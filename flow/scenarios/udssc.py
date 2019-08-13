import xml.etree.ElementTree as ET

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig, TrafficLightParams
from flow.core.util import makexml, printxml, ensure_dir

import os
import time
import numpy as np
import subprocess
from numpy import pi, sin, cos, linspace, sqrt
from lxml import etree
import xml.etree.ElementTree as ElementTree
E = etree.Element
# Number of retries on accessing the .net.xml file before giving up
RETRIES_ON_ERROR = 10
# number of seconds to wait before trying to access the .net.xml file again
WAIT_ON_ERROR = 1


ADDITIONAL_NET_PARAMS = {
    # radius of the loops
    "ring_radius": 50,
    # length of the straight edges connected the outer loop to the inner loop
    "lane_length": 75,
    # length of the merge next to the roundabout
    "merge_length": 15,
    # number of lanes in the inner loop. DEPRECATED. DO NOT USE
    "inner_lanes": 3,
    # number of lanes in the outer loop. DEPRECATED. DO NOT USE
    "outer_lanes": 2, 
    # max speed limit in the roundabout
    "roundabout_speed_limit": 8,
    # max speed limit in the roundabout
    "outside_speed_limit": 15,
    # resolution of the curved portions
    "resolution": 40,
    # num lanes
    "lane_num": 1,
}


class UDSSCMergingScenario(Scenario):
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initializes a two loop scenario where one loop merging in and out of
        the other.

        Requires from net_params:
        - ring_radius: radius of the loops
        - lane_length: length of the straight edges connected the outer loop to
          the inner loop
        - inner_lanes: number of lanes in the inner loop
        - outer_lanes: number of lanes in the outer loop
        - speed_limit: max speed limit in the network
        - resolution: resolution of the curved portions

        See Scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))
        
        self.es = {}

        radius = net_params.additional_params["ring_radius"]
        lane_length = net_params.additional_params["lane_length"]
        merge_length = net_params.additional_params["merge_length"]

        self.junction_length = 0.3
        self.intersection_length = 25.5  # calibrate when the radius changes

        # net_params.additional_params["length"] = \
        #     2 * lane_length + 2 * pi * radius + \
        #     2 * self.intersection_length + 2 * self.junction_length

        self.lane_num = net_params.additional_params["lane_num"]
        
        self._edges, self._connections = self.generate_net(
                net_params,
                traffic_lights,
                self.specify_nodes(net_params),
                self.specify_edges(net_params),
                self.specify_types(net_params),
            )

        super().__init__(name, vehicles, net_params,
                         initial_config, traffic_lights)


    def generate_net(self,
                     net_params,
                     traffic_lights,
                     nodes,
                     edges,
                     types=None,
                     connections=None):
        """Generate Net files for the transportation network.

        Creates different network configuration files for:

        * nodes: x,y position of points which are connected together to form
          links. The nodes may also be fitted with traffic lights, or can be
          treated as priority or zipper merge regions if they combines several
          lanes or edges together.
        * edges: directed edges combining nodes together. These constitute the
          lanes vehicles will be allowed to drive on.
        * types (optional): parameters used to describe common features amount
          several edges of similar types. If edges are not defined with common
          types, this is not needed.
        * connections (optional): describes how incoming and outgoing edge/lane
          pairs on a specific node as connected. If none is specified, SUMO
          handles these connections by default.

        The above files are then combined to form a .net.xml file describing
        the shape of the traffic network in a form compatible with SUMO.

        Parameters
        ----------
        net_params : flow.core.params.NetParams
            network-specific parameters. Different networks require different
            net_params; see the separate sub-classes for more information.
        traffic_lights : flow.core.params.TrafficLightParams
            traffic light information, used to determine which nodes are
            treated as traffic lights
        nodes : list of dict
            A list of node attributes (a separate dict for each node). Nodes
            attributes must include:

            * id {string} -- name of the node
            * x {float} -- x coordinate of the node
            * y {float} -- y coordinate of the node

        edges : list of dict
            A list of edges attributes (a separate dict for each edge). Edge
            attributes must include:

            * id {string} -- name of the edge
            * from {string} -- name of node the directed edge starts from
            * to {string} -- name of the node the directed edge ends at

            In addition, the attributes must contain at least one of the
            following:

            * "numLanes" {int} and "speed" {float} -- the number of lanes and
              speed limit of the edge, respectively
            * type {string} -- a type identifier for the edge, which can be
              used if several edges are supposed to possess the same number of
              lanes, speed limits, etc...

        types : list of dict
            A list of type attributes for specific groups of edges. If none are
            specified, no .typ.xml file is created.
        connections : list of dict
            A list of connection attributes. If none are specified, no .con.xml
            file is created.

        Returns
        -------
        edges : dict <dict>
            Key = name of the edge
            Elements = length, lanes, speed
        connection_data : dict < dict < list < (edge, pos) > > >
            Key = name of the arriving edge
                Key = lane index
                Element = list of edge/lane pairs that a vehicle can traverse
                from the arriving edge/lane pairs
        """
        net_path = os.path.dirname(os.path.abspath(__file__)) \
            + '/debug/net/'
        cfg_path = os.path.dirname(os.path.abspath(__file__)) \
            + '/debug/cfg/'

        ensure_dir('%s' % net_path)
        ensure_dir('%s' % cfg_path)

        network = "temp"
        nodfn = '%s.nod.xml' % network
        edgfn = '%s.edg.xml' % network
        typfn = '%s.typ.xml' % network
        cfgfn = '%s.netccfg' % network
        netfn = '%s.net.xml' % network
        confn = '%s.con.xml' % network
        roufn = '%s.rou.xml' % network
        addfn = '%s.add.xml' % network
        sumfn = '%s.sumo.cfg' % network
        guifn = '%s.gui.cfg' % network
        # self._edges = None
        # self._connections = None
        # self._edge_list = None
        # self._junction_list = None
        # self.__max_speed = None
        self.__length = None
        self.rts = None
        self.cfg = None

        def _import_edges_from_net(net_params):
            """Import edges from a configuration file.
    
            This is a utility function for computing edge information. It imports a
            network configuration file, and returns the information on the edges
            and junctions located in the file.
    
            Parameters
            ----------
            net_params : flow.core.params.NetParams
                see flow/core/params.py
    
            Returns
            -------
            net_data : dict <dict>
                Key = name of the edge/junction
                Element = lanes, speed, length
            connection_data : dict < dict < list < (edge, pos) > > >
                Key = "prev" or "next", indicating coming from or to this
                edge/lane pair
                    Key = name of the edge
                        Key = lane index
                        Element = list of edge/lane pairs preceding or following
                        the edge/lane pairs
            """
            # import the .net.xml file containing all edge/type data
            parser = etree.XMLParser(recover=True)
            net_path = os.path.join(cfg_path, netfn) \
                if net_params.template is None else netfn
            tree = ElementTree.parse(net_path, parser=parser)
            root = tree.getroot()
    
            # Collect information on the available types (if any are available).
            # This may be used when specifying some edge data.
            types_data = dict()
    
            for typ in root.findall('type'):
                type_id = typ.attrib['id']
                types_data[type_id] = dict()
    
                if 'speed' in typ.attrib:
                    types_data[type_id]['speed'] = float(typ.attrib['speed'])
                else:
                    types_data[type_id]['speed'] = None
    
                if 'numLanes' in typ.attrib:
                    types_data[type_id]['numLanes'] = int(typ.attrib['numLanes'])
                else:
                    types_data[type_id]['numLanes'] = None
    
            net_data = dict()
            next_conn_data = dict()  # forward looking connections
            prev_conn_data = dict()  # backward looking connections
    
            # collect all information on the edges and junctions
            for edge in root.findall('edge'):
                edge_id = edge.attrib['id']
    
                # create a new key for this edge
                net_data[edge_id] = dict()
    
                # check for speed
                if 'speed' in edge:
                    net_data[edge_id]['speed'] = float(edge.attrib['speed'])
                else:
                    net_data[edge_id]['speed'] = None
    
                # if the edge has a type parameters, check that type for a
                # speed and parameter if one was not already found
                if 'type' in edge.attrib and edge.attrib['type'] in types_data:
                    if net_data[edge_id]['speed'] is None:
                        net_data[edge_id]['speed'] = \
                            float(types_data[edge.attrib['type']]['speed'])
    
                # collect the length from the lane sub-element in the edge, the
                # number of lanes from the number of lane elements, and if needed,
                # also collect the speed value (assuming it is there)
                net_data[edge_id]['lanes'] = 0
                for i, lane in enumerate(edge):
                    net_data[edge_id]['lanes'] += 1
                    if i == 0:
                        net_data[edge_id]['length'] = float(lane.attrib['length'])
                        if net_data[edge_id]['speed'] is None \
                                and 'speed' in lane.attrib:
                            net_data[edge_id]['speed'] = float(
                                lane.attrib['speed'])
    
                # if no speed value is present anywhere, set it to some default
                if net_data[edge_id]['speed'] is None:
                    net_data[edge_id]['speed'] = 30
    
            # collect connection data
            for connection in root.findall('connection'):
                from_edge = connection.attrib['from']
                from_lane = int(connection.attrib['fromLane'])
    
                if from_edge[0] != ":":
                    # if the edge is not an internal link, then get the next
                    # edge/lane pair from the "via" element
                    via = connection.attrib['via'].rsplit('_', 1)
                    to_edge = via[0]
                    to_lane = int(via[1])
                else:
                    to_edge = connection.attrib['to']
                    to_lane = int(connection.attrib['toLane'])
    
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
    
            connection_data = {'next': next_conn_data, 'prev': prev_conn_data}
    
            return net_data, connection_data

        # add traffic lights to the nodes
        tl_ids = list(traffic_lights.get_properties().keys())
        for n_id in tl_ids:
            indx = next(i for i, nd in enumerate(nodes) if nd['id'] == n_id)
            nodes[indx]['type'] = 'traffic_light'

        # for nodes that have traffic lights that haven't been added
        for node in nodes:
            if node['id'] not in tl_ids \
                    and node.get('type', None) == 'traffic_light':
                traffic_lights.add(node['id'])

            # modify the x and y values to be strings
            node['x'] = str(node['x'])
            node['y'] = str(node['y'])
            if 'radius' in node:
                node['radius'] = str(node['radius'])

        # xml file for nodes; contains nodes for the boundary points with
        # respect to the x and y axes
        x = makexml('nodes', 'http://sumo.dlr.de/xsd/nodes_file.xsd')
        for node_attributes in nodes:
            x.append(E('node', **node_attributes))
        printxml(x, net_path + nodfn)

        # modify the length, shape, numLanes, and speed values
        for edge in edges:
            edge['length'] = str(edge['length'])
            if 'priority' in edge:
                edge['priority'] = str(edge['priority'])
            if 'shape' in edge:
                if not isinstance(edge['shape'], str):
                    edge['shape'] = ' '.join('%.2f,%.2f' % (x, y)
                                             for x, y in edge['shape'])
            if 'numLanes' in edge:
                edge['numLanes'] = str(edge['numLanes'])
            if 'speed' in edge:
                edge['speed'] = str(edge['speed'])

        # xml file for edges
        x = makexml('edges', 'http://sumo.dlr.de/xsd/edges_file.xsd')
        for edge_attributes in edges:
            x.append(E('edge', attrib=edge_attributes))
        printxml(x, net_path + edgfn)

        # xml file for types: contains the the number of lanes and the speed
        # limit for the lanes
        if types is not None:
            # modify the numLanes and speed values
            for typ in types:
                if 'numLanes' in typ:
                    typ['numLanes'] = str(typ['numLanes'])
                if 'speed' in typ:
                    typ['speed'] = str(typ['speed'])

            x = makexml('types', 'http://sumo.dlr.de/xsd/types_file.xsd')
            for type_attributes in types:
                x.append(E('type', **type_attributes))
            printxml(x, net_path + typfn)

        # xml for connections: specifies which lanes connect to which in the
        # edges
        if connections is not None:
            # modify the fromLane and toLane values
            for connection in connections:
                if 'fromLane' in connection:
                    connection['fromLane'] = str(connection['fromLane'])
                if 'toLane' in connection:
                    connection['toLane'] = str(connection['toLane'])

            x = makexml('connections',
                        'http://sumo.dlr.de/xsd/connections_file.xsd')
            for connection_attributes in connections:
                if 'signal_group' in connection_attributes:
                    del connection_attributes['signal_group']
                x.append(E('connection', **connection_attributes))
            printxml(x, net_path + confn)

        # xml file for configuration, which specifies:
        # - the location of all files of interest for sumo
        # - output net file
        # - processing parameters for no internal links and no turnarounds
        x = makexml('configuration',
                    'http://sumo.dlr.de/xsd/netconvertConfiguration.xsd')
        t = E('input')
        t.append(E('node-files', value=nodfn))
        t.append(E('edge-files', value=edgfn))
        if types is not None:
            t.append(E('type-files', value=typfn))
        if connections is not None:
            t.append(E('connection-files', value=confn))
        x.append(t)
        t = E('output')
        t.append(E('output-file', value=netfn))
        x.append(t)
        t = E('processing')
        t.append(E('no-internal-links', value='false'))
        t.append(E('no-turnarounds', value='true'))
        x.append(t)
        printxml(x, net_path + cfgfn)

        subprocess.call(
            [
                'netconvert -c ' + net_path + cfgfn +
                ' --output-file=' + cfg_path + netfn +
                ' --no-internal-links="false"'
            ],
            shell=True)

        # collect data from the generated network configuration file
        error = None
        for _ in range(RETRIES_ON_ERROR):
            try:
                edges_dict, conn_dict = _import_edges_from_net(net_params)
                return edges_dict, conn_dict
            except Exception as e:
                print('Error during start: {}'.format(e))
                print('Retrying in {} seconds...'.format(WAIT_ON_ERROR))
                time.sleep(WAIT_ON_ERROR)
        raise error

    def edge_length(self, edge_id):
        """See parent class."""
        try:
            return self._edges[edge_id]['length']
        except KeyError:
            print('Error in edge length with key', edge_id)
            return -1001

    def specify_edge_starts(self):
        """
        See parent class
        """
        edge_dict = {}
        absolute = 0
        prev_edge = 0
        for edge in self.specify_absolute_order():
            
            if edge.startswith(":"):
                # absolute += float(self.edge_info[edge]["length"])
                absolute += float(self.edge_length(edge))
                continue
            new_x = absolute + prev_edge #+ prev_internal
            edge_dict[edge] = new_x
            # prev_edge = float(self.edge_info[edge]["length"])
            prev_edge = float(self.edge_length(edge))
            absolute = new_x
        self.es.update(edge_dict)

        edgestarts = [ #len of prev edge + total prev (including internal edge len)
            ("right", edge_dict["right"]),
            ("top", edge_dict["top"]), 
            ("left", edge_dict["left"]),
            ("bottom", edge_dict["bottom"]),
            ("inflow_1", edge_dict["inflow_1"]),
            ("merge_in_1", edge_dict["merge_in_1"]),
            ("merge_out_0", edge_dict["merge_out_0"]),
            ("outflow_0", edge_dict["outflow_0"]),
            ("inflow_0", edge_dict["inflow_0"]),
            ("merge_in_0", edge_dict["merge_in_0"]),
            ("merge_out_1", edge_dict["merge_out_1"]),
            ("outflow_1", edge_dict["outflow_1"]),
        ]

        return edgestarts

    def specify_internal_edge_starts(self):
        """
        See parent class
        """
        edge_dict = {}
        absolute = 0
        prev_edge = 0
        for edge in self.specify_absolute_order(): # each edge = absolute + len(prev edge) + len(prev internal edge)
            
            if not edge.startswith(":"):
                # absolute += float(self.edge_info[edge]["length"])
                absolute += float(self.edge_length(edge))
                continue
            new_x = absolute + prev_edge
            edge_dict[edge] = new_x
            # prev_edge = float(self.edge_info[edge]["length"])
            prev_edge = float(self.edge_length(edge))
            absolute = new_x

        if self.lane_num == 2: 
        # two lane 
            internal_edgestarts = [ # in increasing order
                (":a_2", edge_dict[":a_2"]),
                (":b_2", edge_dict[":b_2"]),
                (":c_2", edge_dict[":c_2"]),
                (":d_2", edge_dict[":d_2"]),
                (":g_3", edge_dict[":g_3"]),
                (":b_0", edge_dict[":b_0"]),
                (":e_2", edge_dict[":e_2"]),
                (":e_0", edge_dict[":e_0"]),
                (":d_0", edge_dict[":d_0"]),
                (":g_0", edge_dict[":g_0"]),
            ]
        elif self.lane_num == 1:
        # one lane
            internal_edgestarts = [ # in increasing order
                (":a_1", edge_dict[":a_1"]),
                (":b_1", edge_dict[":b_1"]),
                (":c_1", edge_dict[":c_1"]),
                (":d_1", edge_dict[":d_1"]),
                (":g_2", edge_dict[":g_2"]),
                (":a_0", edge_dict[":a_0"]),
                (":b_0", edge_dict[":b_0"]),
                (":e_1", edge_dict[":e_1"]),
                (":e_0", edge_dict[":e_0"]),
                (":c_0", edge_dict[":c_0"]),
                (":d_0", edge_dict[":d_0"]),
                (":g_0", edge_dict[":g_0"]),
            ]
        self.es.update(edge_dict)

        return internal_edgestarts

    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        """
        See parent class

        Vehicles with the prefix "merge" are placed in the merge ring,
        while all other vehicles are placed in the ring.
        """
        if 'rl_0' in self.vehicles.ids: # HARDCODE ALERT
            # startpositions = [('inflow_0', 10), ('right', 10)]
            startlanes = [0, 0]
            startpositions = [('outflow_0', 0), ('outflow_0', 10)]
        else: 
            startpositions = [('inflow_0', 10)]
            startlanes = [0]
        
        return startpositions, startlanes

    def specify_absolute_order(self):
        if self.lane_num == 2: 
        
            return [":a_2", "right", ":b_2", "top", ":c_2",
                    "left", ":d_2", "bottom", "inflow_1",
                    ":g_3", "merge_in_1", ":a_0", ":b_0",
                    "merge_out_0", ":e_2", "outflow_0", "inflow_0",
                    ":e_0", "merge_in_0", ":c_0", ":d_0",
                    "merge_out_1", ":g_0", "outflow_1" ]
        elif self.lane_num == 1: 
        # one lane
            return [":a_1", "right", ":b_1", "top", ":c_1",
                    "left", ":d_1", "bottom", "inflow_1",
                    ":g_2", "merge_in_1", ":a_0", ":b_0",
                    "merge_out_0", ":e_1", "outflow_0", "inflow_0",
                    ":e_0", "merge_in_0", ":c_0", ":d_0",
                    "merge_out_1", ":g_0", "outflow_1" ]


    def specify_nodes(self, net_params):
        """
        See parent class
        """
        r = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]
        m = net_params.additional_params["merge_length"]

        roundabout_type = "priority"
        default = "priority"

        nodes = [{"id": "a",   "x": repr(0),  "y": repr(-r), "type": roundabout_type},
                 {"id": "b",   "x": repr(0.5 * r),  "y": repr(sqrt(3)/2 * r), "type": roundabout_type},
                 {"id": "c",   "x": repr(-0.5 * r),  "y": repr(sqrt(3)/2 * r), "type": roundabout_type},
                 {"id": "d",   "x": repr(-r), "y": repr(0), "type": roundabout_type},
                 {"id": "e",   "x": repr(0), "y": repr(r + m), "type": default},
                 {"id": "f",   "x": repr(0), "y": repr(r + m + x), "type": default},
                 {"id": "g",   "x": repr(-r - m), "y": repr(-r - 0.1*r), "type": default},
                 {"id": "h",   "x": repr(-r - m - x), "y": repr(-r - 0.2*r), "type": default},
                ]

        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        r = net_params.additional_params["ring_radius"]
        lane_length = net_params.additional_params["lane_length"]
        merge_length = net_params.additional_params["merge_length"]
        circumference = 2 * pi * r
        lanes = repr(net_params.additional_params["lane_num"])
        
        resolution = net_params.additional_params["resolution"]

        circ = 2 * pi * r
        twelfth = circ / 12
        edges = [
            {"id": "bottom",
             "type": "edgeType_hi",
             "from": "d",
             "to": "a",
             "numLanes": lanes,
             "length": repr(twelfth * 3),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(-pi, -pi/2 , resolution)])},

            {"id": "right",
             "type": "edgeType_hi",
             "from": "a",
             "to": "b",
             "numLanes": lanes,
             "length": repr(twelfth * 5),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(-pi / 2,pi/3, resolution)])},

            {"id": "top",
             "type": "edgeType_hi",
             "from": "b",
             "to": "c",
             "numLanes": lanes,
             "length": repr(twelfth * 2),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(pi/3, 2*pi/3, resolution)])},

            {"id": "left",
             "type": "edgeType_hi",
             "from": "c",
             "to": "d", 
             "numLanes": lanes,
             "length": repr(twelfth * 2),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(2*pi/3, pi, resolution)])},

            {"id": "merge_out_0",
             "type": "edgeType_lo",
             "from": "b",
             "to": "e",
             "numLanes": lanes,
             "length": merge_length,
            },

            {"id": "merge_in_0",
             "type": "edgeType_lo",
             "from": "e",
             "to": "c",
             "numLanes": lanes,
             "length": merge_length,
            },

            {"id": "outflow_0",
             "type": "edgeType_lo",
             "from": "e",
             "to": "f",
             "numLanes": lanes,
             "length": lane_length,
            },

            {"id": "inflow_0",
             "type": "edgeType_lo",
             "from": "f",
             "to": "e",
             "numLanes": lanes,
             "length": lane_length,
            },

            {"id": "merge_out_1",
             "type": "edgeType_lo",
             "from": "d",
             "to": "g",
             "numLanes": lanes,
             "length": merge_length,
            },
            
            {"id": "merge_in_1",
             "type": "edgeType_lo",
             "from": "g",
             "to": "a",
             "numLanes": lanes,
             "length": merge_length,
            },

            {"id": "outflow_1",
             "type": "edgeType_lo",
             "from": "g",
             "to": "h",
             "numLanes": lanes,
             "length": lane_length,
            },

            {"id": "inflow_1",
             "type": "edgeType_lo",
             "from": "h",
             "to": "g",
             "numLanes": lanes,
             "length": lane_length,
            },
        ]

        return edges

    def specify_types(self, net_params):
        """
        See parent class
        """
        types = [{"id": "edgeType_hi",
                  "speed": repr(net_params.additional_params.get("roundabout_speed_limit")),
                  "priority": repr(2)},
                 {"id": "edgeType_lo",
                  "speed": repr(net_params.additional_params.get("outside_speed_limit")),
                  "priority": repr(1)}]
        return types

    def specify_routes(self, net_params):
        """
        See parent class
        """

        rts = {"top": {"top": ["top", "left", "bottom", "right"]},
               "left": {"left": ["left", "bottom", "right", "top"]},
               "bottom": {"bottom": ["bottom", "right", "top", "left"]},
               "right": {"right": ["right", "top", "left", "bottom"]},

               "inflow_1": {"inflow_1_0": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"]}, # added
               "inflow_0": {"inflow_1_1": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"]},

            #    "inflow_1": {"inflow_1_0": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"],
            #                 "inflow_1_1": ["inflow_1", "merge_in_1", "right", "merge_out_0", "outflow_0"]}, # added
            #    "inflow_0": {"inflow_0_0": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"],
            #                 "inflow_1_1": ["inflow_0", "merge_in_0", "left", "bottom", "right", "merge_out_0", "outflow_0"]},

               "outflow_1": {"outflow_1": ["outflow_1"]},
               "outflow_0": {"outflow_0": ["outflow_0"]}
               }
               
        rts = {"top": ["top", "left", "bottom", "right"],
               "left": ["left", "bottom", "right", "top"],
               "bottom": ["bottom", "right", "top", "left"],
               "right": ["right", "top", "left", "bottom"],
               "inflow_1": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"], # added
               "inflow_0": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"],

            #    "inflow_1": {"inflow_1_0": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"],
            #                 "inflow_1_1": ["inflow_1", "merge_in_1", "right", "merge_out_0", "outflow_0"]}, # added
            #    "inflow_0": {"inflow_0_0": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"],
            #                 "inflow_1_1": ["inflow_0", "merge_in_0", "left", "bottom", "right", "merge_out_0", "outflow_0"]},

               "outflow_1": ["outflow_1"],
               "outflow_0": ["outflow_0"]
               }
        # routes = Routes()
        # routes.add("top_0", ["top", "left", "bottom", "right"])
        # routes.add("left_0", ["left", "bottom", "right", "top"])
        # routes.add("bottom_0", ["bottom", "right", "top", "left"])
        # routes.add("right_0", ["right", "top", "left", "bottom"])
        # routes.add("inflow_1_0", ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"])
        # # routes.add("inflow_1_1", ["inflow_1", "merge_in_1", "right", "merge_out_0", "outflow_0"])
        # routes.add("inflow_0_0", ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"])
        # # routes.add("inflow_0_1", ["inflow_0", "merge_in_0", "left", "bottom", "right", "merge_out_0", "outflow_0"])
        # routes.add("outflow_1", ["outflow_1"])
        # routes.add("outflow_0", ["outflow_0"])
        # return routes

        return rts