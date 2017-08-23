"""
Base class for generating transportation networks.
"""

from cistar_dev.controllers.base_controller import SumoController
from cistar_dev.core.util import makexml
from cistar_dev.core.util import printxml
from cistar_dev.core.util import ensure_dir

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

    def __init__(self, net_params, net_path, cfg_path, base):
        Serializable.quick_init(self, locals())

        self.net_path = net_path
        self.cfg_path = cfg_path
        self.base = base
        self.name = base
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
        Generates Net files for the transportation network. Different networks require
        different net_params; see the separate sub-classes for more information.
        """
        nodfn = "%s.nod.xml" % self.name
        edgfn = "%s.edg.xml" % self.name
        typfn = "%s.typ.xml" % self.name
        cfgfn = "%s.netccfg" % self.name
        netfn = "%s.net.xml" % self.name
        confn = "%s.con.xml" % self.name

        # specify the attributes of the nodes
        nodes = self.specify_nodes(net_params)

        # xml file for nodes; contains nodes for the boundary points with respect to the x and y axes
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

        # xml file for types: contains the the number of lanes and the speed limit for the lanes
        if types is not None:
            x = makexml("types", "http://sumo.dlr.de/xsd/types_file.xsd")
            for type_attributes in types:
                x.append(E("type", **type_attributes))
            printxml(x, self.net_path + typfn)

        # specify the connection attributes (default is None)
        connections = self.specify_connections(net_params)

        # xml for connections: specifies which lanes connect to which in the edges
        if connections is not None:
            x = makexml("connections", "http://sumo.dlr.de/xsd/connections_file.xsd")
            for connection_attributes in connections:
                x.append(E("connection", **connection_attributes))
            printxml(x, self.net_path + confn)

        # check whether the user requested no-internal-links (default="true")
        if "no-internal-links" not in net_params:
            self.no_internal_links = "true"
        elif net_params["no-internal-links"]:
            self.no_internal_links = "true"
        else:
            self.no_internal_links = "false"

        # xml file for configuration
        # - specifies the location of all files of interest for sumo
        # - specifies output net file
        # - specifies processing parameters for no internal links and no turnarounds
        x = makexml("configuration", "http://sumo.dlr.de/xsd/netconvertConfiguration.xsd")
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
        t.append(E("no-internal-links", value="%s" % self.no_internal_links))
        t.append(E("no-turnarounds", value="true"))
        x.append(t)
        printxml(x, self.net_path + cfgfn)

        retcode = subprocess.call(["netconvert -c " + self.net_path + cfgfn + " --output-file=" +
                                   self.cfg_path + netfn + ' --no-internal-links="%s"' % self.no_internal_links],
                                  stdout=sys.stdout, stderr=sys.stderr, shell=True)
        self.netfn = netfn

        return self.net_path + netfn

    def generate_cfg(self, params):
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
        if "start_time" not in params:
            raise ValueError("start_time not supplied")
        else:
            start_time = params["start_time"]

        if "end_time" in params:
            end_time = params["end_time"]
        else:
            end_time = None

        self.roufn = "%s.rou.xml" % self.name
        addfn = "%s.add.xml" % self.name
        cfgfn = "%s.sumo.cfg" % self.name
        guifn = "%s.gui.cfg" % self.name

        # specify routes vehicles can take
        self.rts = self.specify_routes()

        # TODO: add functionality for multiple routes (do it for Braess)
        add = makexml("additional", "http://sumo.dlr.de/xsd/additional_file.xsd")
        for (edge, route) in self.rts.items():
            add.append(E("route", id="route%s" % edge, edges=" ".join(route)))

        # specify (optional) rerouting actions
        rerouting = self.specify_rerouters()

        if rerouting is not None:
            for rerouting_params in rerouting:
                add.append(self.rerouter(rerouting_params["name"], rerouting_params["from"],
                                         rerouting_params["route"]))

        printxml(add, self.cfg_path + addfn)

        gui = E("viewsettings")
        gui.append(E("scheme", name="real world"))
        printxml(gui, self.cfg_path +guifn)

        cfg = makexml("configuration", "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")

        logging.debug(self.netfn)

        cfg.append(self.inputs(self.name, net=self.netfn, add=addfn, rou=self.roufn, gui=guifn))
        t = E("time")
        t.append(E("begin", value=repr(start_time)))
        if end_time:
            t.append(E("end", value=repr(end_time)))
        cfg.append(t)

        printxml(cfg, self.cfg_path + cfgfn)
        return cfgfn

    def make_routes(self, scenario, initial_config, cfg_params):

        type_params = scenario.type_params
        type_list = [tup[0] for tup in type_params]
        num_cars = scenario.num_vehicles
        if type_list is not None:
            routes = makexml("routes", "http://sumo.dlr.de/xsd/routes_file.xsd")
            for i in range(len(type_list)):
                if type_params[i][2][0] == "sumoIDM":
                    tp = type_params[i][0]

                    # if any IDM parameters are not specified, they are set to the default parameters specified
                    # by Treiber
                    if "accel" not in type_params[tp][1]:
                        type_params[i][2][1]["accel"] = 1

                    if "decel" not in type_params[tp][1]:
                        type_params[i][2][1]["decel"] = 1.5

                    if "delta" not in type_params[tp][1]:
                        type_params[i][2][1]["delta"] = 4

                    if "tau" not in type_params[tp][1]:
                        type_params[i][2][1]["tau"] = 1

                    routes.append(E("vType", attrib={"id": tp, "carFollowModel": "IDM", "minGap": "0",
                                                     "accel": repr(type_params[i][2][1]["accel"]),
                                                     "decel": repr(type_params[i][2][1]["decel"]),
                                                     "delta": repr(type_params[i][2][1]["delta"]),
                                                     "tau": repr(type_params[i][2][1]["tau"])}))
                else:
                    routes.append(E("vType", id=type_params[i][0], minGap="0"))

            self.vehicle_ids = []
            if num_cars > 0:
                for i in range(len(type_params)):
                    tp = type_params[i][0]
                    type_count = type_params[i][1]
                    for j in range(type_count):
                        self.vehicle_ids.append((tp, tp + "_" + str(j)))

            if initial_config["shuffle"]:
                random.shuffle(self.vehicle_ids)

            positions = initial_config["positions"]
            lanes = initial_config["lanes"]
            for i, (veh_type, id) in enumerate(self.vehicle_ids):
                edge, pos = positions[i]
                lane = lanes[i]
                indx_type = [i for i in range(len(type_list)) if type_list[i] == veh_type][0]
                type_depart_speed = type_params[indx_type][4]
                routes.append(self.vehicle(veh_type, "route" + edge, depart="0", id=id, color="1,0.0,0.0",
                                           departSpeed=str(type_depart_speed), departPos=str(pos),
                                           departLane=str(lane)))

            printxml(routes, self.cfg_path + self.roufn)

    def specify_nodes(self, net_params):
        """
        Specifies the attributes of nodes in the network.

        :param net_params: network parameters provided during task initialization
        :return: A list of node attributes (a separate dict for each node). Nodes attributes must include:
                 - id {string} -- name of the node
                 - x {float} -- x coordinate of the node
                 - y {float} -- y coordinate of the node
                 Other attributes may also be specified. See:
                 http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Node_Descriptions
        """
        raise NotImplementedError

    def specify_edges(self, net_params):
        """
        Specifies the attributes of edges containing pairs on nodes in the network.

        :param net_params: network parameters provided during task initialization
        :return: A list of edges attributes (a separate dict for each node). Edge attributes must include:
                 - id {string} -- name of the edge
                 - from {string} -- name of node the directed edge starts from
                 - to {string} -- name of the node the directed edge ends at
                 In addition, the attributes must contain at least on of the following:
                 - "numLanes" {int} and "speed" {float} -- the number of lanes and speed limit
                   of the edge, respectively
                 - type {string} -- a type identifier for the edge, which can be used if several
                   edges are supposed to possess the same number of lanes, speed limits, etc...
                 Other attributes may also be specified. See:
                 http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Edge_Descriptions
        """
        raise NotImplementedError

    def specify_types(self, net_params):
        """
        Specifies the attributes of various edge types (if any exist).

        :param net_params: network parameters provided during task initialization
        :return: A list of type attributes for specific groups of edges. If none are specified,
                 no .typ.xml file is created.
                 For information on type attributes, see:
                 http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Type_Descriptions
        """
        return None

    def specify_connections(self, net_params):
        """
        Specifies the attributes of connections, used to describe how a node's incoming and
        outgoing edges are connected.

        :param net_params: network parameters provided during task initialization
        :return: A list of connection attributes. If none are specified, no .con.xml file is created.
                 For information on type attributes, see:
                 http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Connection_Descriptions
        """
        return None

    def specify_routes(self):
        """
        Specifies the routes vehicles can take starting from a specific node

        :return: a route dict, with the key being the name of the starting edge, and the routes
                 being the edges a vehicle must traverse, starting from the current edge.
        """
        raise NotImplementedError

    def specify_rerouters(self):
        """
        Specifies rerouting actions vehicles should perform once reaching a specific edge.
        """
        return None

    def vtype(self, name, maxSpeed=30, accel=1.5, decel=4.5, length=5, **kwargs):
        return E("vType", accel=repr(accel), decel=repr(decel), id=name, length=repr(length),
                 maxSpeed=repr(maxSpeed), **kwargs)

    def flow(self, name, number, vtype, route, **kwargs):
        return E("flow", id=name, number=repr(number), route=route, type=vtype, **kwargs)

    def vehicle(self, type, route, departPos, number=0, id=None, **kwargs):
        if not id and not number:
            raise ValueError("Supply either ID or Number")
        if not id:
            id = type + "_" + str(number)
        return E("vehicle", type=type, id=id, route=route, departPos=departPos, **kwargs)

    def inputs(self, name, net=None, rou=None, add=None, gui=None):
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

    def rerouter(self, name, frm, to):
        t = E("rerouter", id=name, edges=frm)
        i = E("interval", begin="0", end="10000000")
        i.append(E("routeProbReroute", id=to))
        t.append(i)
        return t
