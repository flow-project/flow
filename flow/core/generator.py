"""
Base class for generating transportation networks.
"""
from flow.core.util import makexml
from flow.core.util import printxml
from flow.core.util import ensure_dir
from flow.controllers.base_controller import SumoController

import sys
import subprocess
import logging
import random
from lxml import etree

from rllab.core.serializable import Serializable

E = etree.Element


class Generator(Serializable):
    CFG_PATH = "./"
    NET_PATH = "./"

    def __init__(self, net_params, base):
        Serializable.quick_init(self, locals())

        self.net_params = net_params
        self.net_path = net_params.net_path
        self.cfg_path = net_params.cfg_path
        self.base = base
        self.netfn = ""
        self.vehicle_ids = []

        ensure_dir("%s" % self.net_path)
        ensure_dir("%s" % self.cfg_path)

        # if a name was not specified by the sub-class's initialization,
        # use the base as the name
        if not hasattr(self, "name"):
            self.name = "%s" % self.base

    def generate_net(self, net_params):
        """
        Generates Net files for the transportation network. Different networks
        require different net_params; see the separate sub-classes for more
        information.
        """
        nodfn = "%s.nod.xml" % self.name
        edgfn = "%s.edg.xml" % self.name
        typfn = "%s.typ.xml" % self.name
        cfgfn = "%s.netccfg" % self.name
        netfn = "%s.net.xml" % self.name
        confn = "%s.con.xml" % self.name

        # specify the attributes of the nodes
        nodes = self.specify_nodes(net_params)

        # xml file for nodes; contains nodes for the boundary points with
        # respect to the x and y axes
        x = makexml("nodes", "http://sumo.dlr.de/xsd/nodes_file.xsd")
        for node_attributes in nodes:
            x.append(E("node", **node_attributes))
        printxml(x, self.net_path + nodfn)

        # collect the attributes of each edge
        edges = self.specify_edges(net_params)

        # xml file for edges
        x = makexml("edges", "http://sumo.dlr.de/xsd/edges_file.xsd")
        for edge_attributes in edges:
            x.append(E("edge", attrib=edge_attributes))
        printxml(x, self.net_path + edgfn)

        # specify the types attributes (default is None)
        types = self.specify_types(net_params)

        # xml file for types: contains the the number of lanes and the speed
        # limit for the lanes
        if types is not None:
            x = makexml("types", "http://sumo.dlr.de/xsd/types_file.xsd")
            for type_attributes in types:
                x.append(E("type", **type_attributes))
            printxml(x, self.net_path + typfn)

        # specify the connection attributes (default is None)
        connections = self.specify_connections(net_params)

        # xml for connections: specifies which lanes connect to which in the
        # edges
        if connections is not None:
            x = makexml("connections",
                        "http://sumo.dlr.de/xsd/connections_file.xsd")
            for connection_attributes in connections:
                x.append(E("connection", **connection_attributes))
            printxml(x, self.net_path + confn)

        # check whether the user requested no-internal-links (default="true")
        if net_params.no_internal_links:
            no_internal_links = "true"
        else:
            no_internal_links = "false"

        # xml file for configuration, which specifies:
        # - the location of all files of interest for sumo
        # - output net file
        # - processing parameters for no internal links and no turnarounds
        x = makexml("configuration",
                    "http://sumo.dlr.de/xsd/netconvertConfiguration.xsd")
        t = E("input")
        t.append(E("node-files", value=nodfn))
        t.append(E("edge-files", value=edgfn))
        if types is not None:
            t.append(E("type-files", value=typfn))
        if connections is not None:
            t.append(E("connection-files", value=confn))
        x.append(t)
        t = E("output")
        t.append(E("output-file", value=netfn))
        x.append(t)
        t = E("processing")
        t.append(E("no-internal-links", value="%s" % no_internal_links))
        t.append(E("no-turnarounds", value="true"))
        x.append(t)
        printxml(x, self.net_path + cfgfn)

        retcode = subprocess.call(
            ["netconvert -c " + self.net_path + cfgfn + " --output-file=" +
             self.cfg_path + netfn + ' --no-internal-links="%s"'
             % no_internal_links],
            stdout=sys.stdout, stderr=sys.stderr, shell=True)

        self.netfn = netfn

        return self.net_path + netfn

    def generate_cfg(self, net_params):
        """
        Generates .sumo.cfg files using net files and netconvert.
        Requires:
        num_cars: Number of cars to seed the simulation with
           max_speed: max speed of cars
           OR
        type_list: List of types of cars to seed the simulation with

        start_time: time to start the simulation
        end_time: time to end the simulation
        """
        start_time = 0
        end_time = None

        self.roufn = "%s.rou.xml" % self.name
        addfn = "%s.add.xml" % self.name
        cfgfn = "%s.sumo.cfg" % self.name
        guifn = "%s.gui.cfg" % self.name

        # specify routes vehicles can take
        self.rts = self.specify_routes(net_params)

        # TODO: add functionality for multiple routes (such as Braess)
        add = makexml("additional",
                      "http://sumo.dlr.de/xsd/additional_file.xsd")
        for (edge, route) in self.rts.items():
            add.append(E("route", id="route%s" % edge, edges=" ".join(route)))

        # specify (optional) rerouting actions
        rerouting = self.specify_rerouters(net_params)

        if rerouting is not None:
            for rerouting_params in rerouting:
                add.append(self._rerouter(rerouting_params["name"],
                                          rerouting_params["from"],
                                          rerouting_params["route"]))

        printxml(add, self.cfg_path + addfn)

        gui = E("viewsettings")
        gui.append(E("scheme", name="real world"))
        printxml(gui, self.cfg_path +guifn)

        cfg = makexml("configuration",
                      "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")

        logging.debug(self.netfn)

        cfg.append(self._inputs(self.name, net=self.netfn, add=addfn,
                                rou=self.roufn, gui=guifn))
        t = E("time")
        t.append(E("begin", value=repr(start_time)))
        if end_time:
            t.append(E("end", value=repr(end_time)))
        cfg.append(t)

        printxml(cfg, self.cfg_path + cfgfn)
        return cfgfn

    def make_routes(self, scenario, initial_config):

        vehicles = scenario.vehicles
        if vehicles.num_vehicles > 0:
            routes = makexml("routes", "http://sumo.dlr.de/xsd/routes_file.xsd")

            # add the types of vehicles to the xml file
            for veh_type in vehicles.types:
                # find a vehicle with this type, and collect its acceleration
                # controller
                for veh_id in vehicles.get_ids():
                    if vehicles.get_state(veh_id, "type") == veh_type:
                        acc_controller = vehicles.get_acc_controller(veh_id)
                        break

                # check if the vehicle type uses SumoController
                if type(acc_controller) == SumoController:
                    # adopt the parameters specified by the SumoController
                    contr_params = acc_controller.controller_params
                    for key in contr_params.keys():
                        contr_params[key] = str(contr_params[key])

                    routes.append(E("vType", id=veh_type, **contr_params))
                else:
                    # default vehicle parameters, intended to provide the user
                    # with a high level of control over the vehicle
                    routes.append(E("vType", id=veh_type, minGap="0",
                                    accel="100", decel="100"))

            self.vehicle_ids = vehicles.get_ids()

            if initial_config.shuffle:
                random.shuffle(self.vehicle_ids)

            # add the initial positions of vehicles to the xml file
            positions = initial_config.positions
            lanes = initial_config.lanes
            for i, id in enumerate(self.vehicle_ids):
                veh_type = vehicles.get_state(id, "type")
                edge, pos = positions[i]
                lane = lanes[i]
                type_depart_speed = vehicles.get_initial_speed(id)
                routes.append(self._vehicle(
                    veh_type, "route" + edge, depart="0", id=id,
                    color="1,0.0,0.0", departSpeed=str(type_depart_speed),
                    departPos=str(pos), departLane=str(lane)))

            printxml(routes, self.cfg_path + self.roufn)

    def specify_nodes(self, net_params):
        """
        Specifies the attributes of nodes in the network.

        Parameters
        ----------
        net_params: NetParams type
            see flow/core/params.py

        Returns
        -------
        nodes: list of dict
            A list of node attributes (a separate dict for each node). Nodes
            attributes must include:
             - id {string} -- name of the node
             - x {float} -- x coordinate of the node
             - y {float} -- y coordinate of the node

        Other attributes may also be specified. See:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Node_Descriptions
        """
        raise NotImplementedError

    def specify_edges(self, net_params):
        """
        Specifies the attributes of edges containing pairs on nodes in the
        network.

        Parameters
        ----------
        net_params: NetParams type
            see flow/core/params.py

        Returns
        -------
        edges: list of dict
            A list of edges attributes (a separate dict for each edge). Edge
            attributes must include:
             - id {string} -- name of the edge
             - from {string} -- name of node the directed edge starts from
             - to {string} -- name of the node the directed edge ends at
            In addition, the attributes must contain at least one of the
            following:
             - "numLanes" {int} and "speed" {float} -- the number of lanes and
               speed limit of the edge, respectively
             - type {string} -- a type identifier for the edge, which can be
               used if several edges are supposed to possess the same number of
               lanes, speed limits, etc...

        Other attributes may also be specified. See:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Edge_Descriptions
        """
        raise NotImplementedError

    def specify_types(self, net_params):
        """
        Specifies the attributes of various edge types (if any exist).

        Parameters
        ----------
        net_params: NetParams type
            see flow/core/params.py

        Returns
        -------
        types: list of dict
            A list of type attributes for specific groups of edges. If none are
            specified, no .typ.xml file is created.

        For information on type attributes, see:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Type_Descriptions
        """
        return None

    def specify_connections(self, net_params):
        """
        Specifies the attributes of connections, used to describe how a node's
        incoming and outgoing edges are connected.

        Parameters
        ----------
        net_params: NetParams type
            see flow/core/params.py

        Returns
        -------
        connections: list of dict
            A list of connection attributes. If none are specified, no .con.xml
            file is created.

        For information on type attributes, see:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Connection_Descriptions
        """
        return None

    def specify_routes(self, net_params):
        """
        Specifies the routes vehicles can take starting from a specific node

        Parameters
        ----------
        net_params: NetParams type
            see flow/core/params.py

        Returns
        -------
        routes: dict
            Key = name of the starting edge
            Element = list of edges a vehicle starting from this edge must
            traverse.
        """
        raise NotImplementedError

    def specify_rerouters(self, net_params):
        """
        Specifies rerouting actions vehicles should perform once reaching a
        specific edge.

        Parameters
        ----------
        net_params: NetParams type
            see flow/core/params.py

        Returns
        -------
        rerouters: list of dict
            A list of rerouting attributes (a separate dict for each rerouter),
            with each dict containing:
             - name {string} -- name of the rerouter
             - from {string} -- the edge in which rerouting takes place
             - route {string} -- name of the route the vehicle is rerouted into
        """
        return None

    def _vtype(self, name, maxSpeed=30, accel=1.5, decel=4.5, length=5,
               **kwargs):
        return E("vType", accel=repr(accel), decel=repr(decel), id=name,
                 length=repr(length), maxSpeed=repr(maxSpeed), **kwargs)

    def _flow(self, name, number, vtype, route, **kwargs):
        return E("flow", id=name, number=repr(number), route=route, type=vtype,
                 **kwargs)

    def _vehicle(self, type, route, departPos, number=0, id=None, **kwargs):
        if not id and not number:
            raise ValueError("Supply either ID or Number")
        if not id:
            id = type + "_" + str(number)
        return E("vehicle", type=type, id=id, route=route, departPos=departPos,
                 **kwargs)

    def _inputs(self, name, net=None, rou=None, add=None, gui=None):
        inp = E("input")
        if net is not False:
            if net is None:
                inp.append(E("net-file", value="%s.net.xml" % name))
            else:
                inp.append(E("net-file", value=net))
        if rou is not False:
            if rou is None:
                inp.append(E("route-files", value="%s.rou.xml" % name))
            else:
                inp.append(E("route-files", value=rou))
        if add is not False:
            if add is None:
                inp.append(E("additional-files", value="%s.add.xml" % name))
            else:
                inp.append(E("additional-files", value=add))
        if gui is not False:
            if gui is None:
                inp.append(E("gui-settings-file", value="%s.gui.xml" % name))
            else:
                inp.append(E("gui-settings-file", value=gui))
        return inp

    def _rerouter(self, name, frm, to):
        t = E("rerouter", id=name, edges=frm)
        i = E("interval", begin="0", end="10000000")
        i.append(E("routeProbReroute", id=to))
        t.append(i)
        return t
