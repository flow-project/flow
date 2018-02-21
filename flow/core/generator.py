from flow.core.util import makexml, printxml, ensure_dir

import subprocess
import logging
import random
import os
from lxml import etree
import xml.etree.ElementTree as ET

try:
    # Import serializable if rllab is installed
    from rllab.core.serializable import Serializable
except ImportError as e:
    Serializable = object

E = etree.Element


class Generator(Serializable):

    def __init__(self, net_params, base):
        """
        Base class for generating transportation networks.
        """
        # Invoke serializable if using rllab
        if Serializable is not object:
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

    def generate_net(self, net_params, traffic_lights):
        """
        Generates Net files for the transportation network. Different networks
        require different net_params; see the separate sub-classes for more
        information.

        Parameters
        ----------
        net_params : flow.core.params.NetParams type
            network-specific parameters
        traffic_lights : flow.core.traffic_lights.TrafficLights type
            traffic light information

        Returns
        -------
        edges : dict <dict>
            Key = name of the edge
            Elements = length, lanes, speed
        connection_data : dict < dict < list<tup> > >
            Key = name of the arriving edge
                Key = lane index
                Element = list of edge/lane pairs that a vehicle can traverse
                from the arriving edge/lane pairs
        """
        nodfn = "%s.nod.xml" % self.name
        edgfn = "%s.edg.xml" % self.name
        typfn = "%s.typ.xml" % self.name
        cfgfn = "%s.netccfg" % self.name
        netfn = "%s.net.xml" % self.name
        confn = "%s.con.xml" % self.name

        # specify the attributes of the nodes
        nodes = self.specify_nodes(net_params)

        # add traffic lights to the nodes
        for n_id in traffic_lights.get_ids():
            print(nodes)
            print(n_id)
            indx = next(i for i, node in enumerate(nodes) if node["id"] == n_id)
            nodes[indx]["type"] = "traffic_light"

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

        fh = open(self.cfg_path + "NULL", "w")
        subprocess.call(
            ["netconvert -c " + self.net_path + cfgfn + " --output-file=" +
             self.cfg_path + netfn + ' --no-internal-links="%s"'
             % no_internal_links],
            stdout=fh, stderr=fh, shell=True)
        fh.close()

        # location of the .net.xml file
        self.netfn = netfn

        # collect data from the generated network configuration file
        edges_dict, conn_dict = self._import_edges_from_net()

        return edges_dict, conn_dict

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
            for vtype, type_params in vehicles.types:
                type_params_str = {key: str(type_params[key])
                                   for key in type_params}
                routes.append(E("vType", id=vtype, **type_params_str))

            self.vehicle_ids = vehicles.get_ids()

            if initial_config.shuffle:
                random.shuffle(self.vehicle_ids)

            # add the initial positions of vehicles to the xml file
            positions = initial_config.positions
            lanes = initial_config.lanes
            for i, veh_id in enumerate(self.vehicle_ids):
                veh_type = vehicles.get_state(veh_id, "type")
                edge, pos = positions[i]
                lane = lanes[i]
                type_depart_speed = vehicles.get_initial_speed(veh_id)
                routes.append(self._vehicle(
                    veh_type, "route" + edge, depart="0", id=veh_id,
                    color="1,0.0,0.0", departSpeed=str(type_depart_speed),
                    departPos=str(pos), departLane=str(lane)))

            # add the in-flows from various edges to the xml file
            if self.net_params.in_flows is not None:
                total_inflows = self.net_params.in_flows.get()
                for inflow in total_inflows:
                    for key in inflow:
                        if not isinstance(inflow[key], str):
                            inflow[key] = repr(inflow[key])
                    routes.append(self._flow(**inflow))

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

    def _flow(self, name, vtype, route, **kwargs):
        return E("flow", id=name, route=route, type=vtype, **kwargs)

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

    def _import_edges_from_net(self):
        """
        Imports a network configuration file, and returns the information on the
        edges and junctions located in the file.

        Return
        ------
        net_data : dict <dict>
            Key = name of the edge/junction
            Element = lanes, speed, length
        connection_data : dict < dict < dict < list<tup> > > >
            Key = "prev" or "next", indicating coming from or to this
            edge/lane pair
                Key = name of the edge
                    Key = lane index
                    Element = list of edge/lane pairs preceding or following
                    the edge/lane pairs
        """
        # import the .net.xml file containing all edge/type data
        parser = etree.XMLParser(recover=True)
        tree = ET.parse(os.path.join(self.net_params.cfg_path, self.netfn),
                        parser=parser)
        root = tree.getroot()

        # Collect information on the available types (if any are available).
        # This may be used when specifying some edge data.
        types_data = dict()

        for typ in root.findall('type'):
            type_id = typ.attrib["id"]
            types_data[type_id] = dict()

            if "speed" in typ.attrib:
                types_data[type_id]["speed"] = float(typ.attrib["speed"])
            else:
                types_data[type_id]["speed"] = None

            if "numLanes" in typ.attrib:
                types_data[type_id]["numLanes"] = int(typ.attrib["numLanes"])
            else:
                types_data[type_id]["numLanes"] = None

        net_data = dict()
        next_conn_data = dict()  # forward looking connections
        prev_conn_data = dict()  # backward looking connections

        # collect all information on the edges and junctions
        for edge in root.findall('edge'):
            edge_id = edge.attrib["id"]

            # create a new key for this edge
            net_data[edge_id] = dict()

            # check for speed
            if "speed" in edge:
                net_data[edge_id]["speed"] = float(edge.attrib["speed"])
            else:
                net_data[edge_id]["speed"] = None

            # if the edge has a type parameters, check that type for a speed and
            # parameter if one was not already found
            if "type" in edge.attrib and edge.attrib["type"] in types_data:
                if net_data[edge_id]["speed"] is None:
                    net_data[edge_id]["speed"] = \
                        float(types_data[edge.attrib["type"]]["speed"])

            # collect the length from the lane sub-element in the edge, the
            # number of lanes from the number of lane elements, and if needed,
            # also collect the speed value (assuming it is there)
            net_data[edge_id]["lanes"] = 0
            for i, lane in enumerate(edge):
                net_data[edge_id]["lanes"] += 1
                if i == 0:
                    net_data[edge_id]["length"] = float(lane.attrib["length"])
                    if net_data[edge_id]["speed"] is None \
                            and "speed" in lane.attrib:
                        net_data[edge_id]["speed"] = float(lane.attrib["speed"])

            # if no speed value is present anywhere, set it to some default
            if net_data[edge_id]["speed"] is None:
                net_data[edge_id]["speed"] = 30

        # collect connection data
        for connection in root.findall('connection'):
            from_edge = connection.attrib["from"]
            from_lane = int(connection.attrib["fromLane"])

            if from_edge[0] != ":" and not self.net_params.no_internal_links:
                # if the edge is not an internal links and the network is
                # allowed to have internal links, then get the next edge/lane
                # pair from the "via" element
                via = connection.attrib["via"].rsplit("_", 1)
                to_edge = via[0]
                to_lane = int(via[1])
            else:
                to_edge = connection.attrib["to"]
                to_lane = int(connection.attrib["toLane"])

            if from_edge not in next_conn_data:
                next_conn_data[from_edge] = dict()

            if from_lane not in next_conn_data[from_edge]:
                next_conn_data[from_edge][from_lane] = list()

            if to_edge not in prev_conn_data:
                prev_conn_data[to_edge] = dict()

            if to_lane not in prev_conn_data[to_edge]:
                prev_conn_data[to_edge][to_lane] = list()

            next_conn_data[from_edge][from_lane].append((to_edge, to_lane))
            prev_conn_data[to_edge][to_lane].append((from_edge, from_lane))

        connection_data = {"next": next_conn_data, "prev": prev_conn_data}

        return net_data, connection_data
