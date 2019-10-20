"""Contains the traffic light grid scenario class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from collections import defaultdict
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # dictionary of traffic light grid array data
    "grid_array": {
        # number of horizontal rows of edges
        "row_num": 3,
        # number of vertical columns of edges
        "col_num": 2,
        # length of edges in the traffic light grid network
        "inner_length": None,
        # length of edges where vehicles enter the network
        "short_length": None,
        # length of edges where vehicles exit the network
        "long_length": None,
        # number of cars starting at the edges heading to the top
        "cars_top": 20,
        # number of cars starting at the edges heading to the bottom
        "cars_bot": 20,
        # number of cars starting at the edges heading to the left
        "cars_left": 20,
        # number of cars starting at the edges heading to the right
        "cars_right": 20,
    },
    # number of lanes in the horizontal edges
    "horizontal_lanes": 1,
    # number of lanes in the vertical edges
    "vertical_lanes": 1,
    # speed limit for all edges, may be represented as a float value, or a
    # dictionary with separate values for vertical and horizontal lanes
    "speed_limit": {
        "horizontal": 35,
        "vertical": 35
    }
}


class TrafficLightGridNetwork(Network):
    """Traffic Light Grid network class.
    The traffic light grid network consists of m vertical lanes and n
    horizontal lanes, with a total of nxm intersections where the vertical
    and horizontal edges meet.
    Requires from net_params:
    * **grid_array** : dictionary of grid array data, with the following keys
      * **row_num** : number of horizontal rows of edges
      * **col_num** : number of vertical columns of edges
      * **inner_length** : length of inner edges in traffic light grid network
      * **short_length** : length of edges that vehicles start on
      * **long_length** : length of final edge in route
      * **cars_top** : number of cars starting at the edges heading to the top
      * **cars_bot** : number of cars starting at the edges heading to the
        bottom
      * **cars_left** : number of cars starting at the edges heading to the
        left
      * **cars_right** : number of cars starting at the edges heading to the
        right
    * **horizontal_lanes** : number of lanes in the horizontal edges
    * **vertical_lanes** : number of lanes in the vertical edges
    * **speed_limit** : speed limit for all edges. This may be represented as a
      float value, or a dictionary with separate values for vertical and
      horizontal lanes.
    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import TrafficLightGridNetwork
    >>>
    >>> network = TrafficLightGridNetwork(
    >>>     name='grid',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'grid_array': {
    >>>                 'row_num': 3,
    >>>                 'col_num': 2,
    >>>                 'inner_length': 500,
    >>>                 'short_length': 500,
    >>>                 'long_length': 500,
    >>>                 'cars_top': 20,
    >>>                 'cars_bot': 20,
    >>>                 'cars_left': 20,
    >>>                 'cars_right': 20,
    >>>             },
    >>>             'horizontal_lanes': 1,
    >>>             'vertical_lanes': 1,
    >>>             'speed_limit': {
    >>>                 'vertical': 35,
    >>>                 'horizontal': 35
    >>>             }
    >>>         },
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize an n*m traffic light grid network."""
        optional = ["tl_logic"]
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params and p not in optional:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        for p in ADDITIONAL_NET_PARAMS["grid_array"].keys():
            if p not in net_params.additional_params["grid_array"]:
                raise KeyError(
                    'Grid array parameter "{}" not supplied'.format(p))

        # retrieve all additional parameters
        # refer to the ADDITIONAL_NET_PARAMS dict for more documentation
        self.vertical_lanes = net_params.additional_params["vertical_lanes"]
        self.horizontal_lanes = net_params.additional_params[
            "horizontal_lanes"]
        self.speed_limit = net_params.additional_params["speed_limit"]
        if not isinstance(self.speed_limit, dict):
            self.speed_limit = {
                "horizontal": self.speed_limit,
                "vertical": self.speed_limit
            }

        self.grid_array = net_params.additional_params["grid_array"]
        self.row_num = self.grid_array["row_num"]
        self.col_num = self.grid_array["col_num"]
        self.inner_length = self.grid_array["inner_length"]
        self.short_length = self.grid_array["short_length"]
        self.long_length = self.grid_array["long_length"]
        self.cars_heading_top = self.grid_array["cars_top"]
        self.cars_heading_bot = self.grid_array["cars_bot"]
        self.cars_heading_left = self.grid_array["cars_left"]
        self.cars_heading_right = self.grid_array["cars_right"]

        # specifies whether or not there will be traffic lights at the
        # intersections (True by default)
        self.use_traffic_lights = net_params.additional_params.get(
            "traffic_lights", True)

        # radius of the inner nodes (ie of the intersections)
        self.nodes_radius = 2.9 + 3.3 * max(self.vertical_lanes,
                                            self.horizontal_lanes)

        # total number of edges in the network
        self.num_edges = 4 * ((self.col_num + 1) * self.row_num + self.col_num)

        # name of the network (DO NOT CHANGE)
        self.name = "BobLoblawsLawBlog"

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        return self._nodes

    def specify_edges(self, net_params):
        """See parent class."""
        return self._edges

    def specify_routes(self, net_params):
        """See parent class."""
        routes = defaultdict(list)

        # build row routes (vehicles go from left to right and vice versa)
        for i in range(self.row_num):
            bot_id = "bot{}_0".format(i)
            top_id = "top{}_{}".format(i, self.col_num)
            for j in range(self.col_num + 1):
                routes[bot_id] += ["bot{}_{}".format(i, j)]
                routes[top_id] += ["top{}_{}".format(i, self.col_num - j)]

        # build column routes (vehicles go from top to bottom and vice versa)
        for j in range(self.col_num):
            left_id = "left{}_{}".format(self.row_num, j)
            right_id = "right0_{}".format(j)
            for i in range(self.row_num + 1):
                routes[left_id] += ["left{}_{}".format(self.row_num - i, j)]
                routes[right_id] += ["right{}_{}".format(i, j)]

        routes["bot0_0"] = ["bot0_0", "right1_0"]
        routes["bot1_0"] = ["bot1_0", "right2_0"]

        return routes

    def specify_types(self, net_params):
        """See parent class."""
        types = [{
            "id": "horizontal",
            "numLanes": self.horizontal_lanes,
            "speed": self.speed_limit["horizontal"]
        }, {
            "id": "vertical",
            "numLanes": self.vertical_lanes,
            "speed": self.speed_limit["vertical"]
        }]

        return types

    # ===============================
    # ============ UTILS ============
    # ===============================

    @property
    def _nodes(self):
        """Build out the nodes of the network.
        The nodes correspond to the intersections between the roads and the starting and ending positions of edges.
        The labeled according to their position in an x-y plane, with the bottom left corner being node "(0.0)"

        For example, the nodes in a traffic light grid with 2 rows and 3 columns
        would be indexed as follows:

               ^ y
               |
            3  -          (1.3)     (2.3)     (3.3)
               |            |         |         |
               |            |         |         |
            2  -(0.2) --- (1.2) --- (2.2) --- (3.2) --- (4.2)
               |            |         |         |
               |            |         |         |
            1  -(0.1) --- (1.1) --- (2.1) --- (3.1) --- (4.1)
               |            |         |         |
               |            |         |         |
            0  -          (1.0)     (2.0)     (3.0)
               |
               ----|--------|---------|---------|---------|---> x
                   0        1         2         3         4

        The id of a node is then "({x}.{y})", for instance "(1.0)" for
        the node at x,y coordinates of (1,0). Note that we're taking the bottom
        left corner as the origin, i.e. (0,0). Also, note that we're using a dot "."
        instead of a comma "," to separate the x and y co-ordinate in the node ids.

        Returns
        -------
        list <dict>
            List of all the nodes in the network
        """

        node_type = "traffic_light" if self.use_traffic_lights else "priority"
        x_max = self.col_num + 1
        y_max = self.row_num + 1

        nodes = []
        for x in range(x_max + 1):
            for y in range(y_max + 1):
                if (x, y) not in [(0, 0), (x_max, 0), (0, y_max), (x_max, y_max)]:
                    nodes.append({
                        "id": "({}.{})".format(x, y),
                        "x": x * self.inner_length,
                        "y": y * self.inner_length,
                        "type": node_type,
                        "radius": self.nodes_radius
                    })

        return nodes

    @property
    def _tl_nodes(self):
        return self._nodes

    @property
    def _edges(self):
        """Build out the edges of the network.
        Edges join nodes to each other.
        Consider the following network with n = 2 rows and m = 3 columns,
        where the nodes are marked by 'x':

        y
        ^
        |
    3   -         x     x     x
        |         |     |     |
    2   -    x----x-----x-----x----x
        |         |     |     |
    1   -    x----x-----x-(*)-x----x
        |         |     |     |
    0   -         x     x     x
        |
        ----|-----|-----|-----|----|----> x
             0    1     2     3    4


        There are n * (m + 1) = 8 horizontal edges and m * (n + 1) = 9
        vertical edges, all that multiplied by two because each edge
        consists of two roads going in opposite directions traffic-wise.
        Edge ids take the format of "({from_node})--({to_node})".

        For example, on edge (*) the id of the bottom road (traffic
        going from left to right) is "(2.1)--(3.1)" and the id of the top road
        (traffic going from right to left) is "(3.1)--(2.1)".
        Returns
        -------
        list <dict>
            List of inner edges
        """
        edges = []

        x_max = self.col_num + 1
        y_max = self.row_num + 1

        def new_edge(from_node, to_node, orientation):
            return [{
                "id": str(from_node) + "--" + str(to_node),
                "type": orientation,
                "priority": 78,
                "from": str(from_node),
                "to": str(to_node),
                "length": self.inner_length
            }]

        # Build the horizontal edges
        for y in range(1, y_max):
            for x in range(x_max):
                left_node = "({}.{})".format(x, y)
                right_node = "({}.{})".format(x + 1, y)
                edges += new_edge(left_node, right_node,
                                  "horizontal")
                edges += new_edge(right_node, left_node,
                                  "horizontal")
        # Build the vertical edges
        for x in range(1, x_max):
            for y in range(y_max):
                bottom_node = "({}.{})".format(x, y)
                top_node = "({}.{})".format(x, y + 1)
                edges += new_edge(bottom_node, top_node,
                                  "vertical")
                edges += new_edge(top_node, bottom_node,
                                  "vertical")

        return edges

    def specify_connections(self, net_params):
        """Build out connections at each node (aside from those nodes where vehicles enter and exit).
        Connections describe what happens at the intersections. We specify the connections of an entire network using
        a dict, where keys are the individual node ids and the values are a list of all connections directly related to
        a particular node. For a particular node, we specify the node's connections using a list of "from" edges and "to"
        edges. Here, we link lanes with all possible adjacent lanes. This means vehicles can make
        turns at any intersection and can turn to any particular adjacent lane.

        Returns
        -------
        dict<list<dict>>
            Dict of all the connections in the network
        """
        con_dict = {}
        x_max = self.col_num + 1
        y_max = self.row_num + 1

        # def new_con(side, from_id, to_id, lane, signal_group):
        #     return [{
        #         "from": side + from_id,
        #         "to": side + to_id,
        #         "fromLane": str(lane),
        #         "toLane": str(lane),
        #         "signal_group": signal_group
        #     }]

        def node_cons(row, col, signal_group):
            """Build all 12 connections for a particular inner node. An inner node
            has 4 edges entering and 4 edges leaving it. Returns a list of 12 dicts"""
            connections = []

            def single_con(origin, dest, lane_num, signal_group_num):
                """Build a single connection given origin and destination edges and lanes ,
                plus a signal group. Returns a list  with a single dict."""
                return [{
                    "from": origin,
                    "to": dest,
                    "fromLane": str(lane_num),
                    "toLane": str(lane_num),
                    "signal_group": signal_group_num
                }]

            def triple_cons(num_lanes, origin, left, straight, right, signal_group):
                """"Connect the origin edge to the eft, straight and right edges.
                Returns a list with 3 * num_lanes"""
                cons = []
                for lane in range(num_lanes):
                    cons += single_con(origin, left, lane, signal_group)  # left turns
                    cons += single_con(origin, straight, lane, signal_group)  # straight turns
                    cons += single_con(origin, right, lane, signal_group)  # right turns

                return cons

            def build_bot_conns(rr, cc):
                """Build the left, straight and right connections for the bottom edge
                entering the node. Returns a list of three dicts"""

                origin = "bot" + "{}_{}".format(rr, cc)
                dest_left = "right" + "{}_{}".format(rr + 1, cc)
                dest_straight = "bot" + "{}_{}".format(rr, cc + 1)
                dest_right = "left" + "{}_{}".format(rr, cc)

                # making assumption number of vertical_lanes = number of horizontal lanes

                return triple_cons(self.vertical_lanes, origin, dest_left, dest_straight, dest_right, signal_group)

            def build_left_conns(rr, cc):
                """Build the left, straight and right connections for the left horizontal edge
                entering the node. Returns a list of three dicts"""

                origin = "left" + "{}_{}".format(rr + 1, cc)
                dest_left = "bot" + "{}_{}".format(rr, cc + 1)
                dest_straight = "left" + "{}_{}".format(rr, cc)
                dest_right = "top" + "{}_{}".format(rr, cc)

                # Assumption: number of vertical_lanes = number of horizontal lanes

                return triple_cons(self.vertical_lanes, origin, dest_left, dest_straight, dest_right, signal_group)

            def build_top_conns(rr, cc):
                """Build the left, straight and right connections for the left horizontal edge
                entering the node. Returns a list of three dicts"""

                origin = "top" + "{}_{}".format(rr, cc + 1)
                dest_left = "left" + "{}_{}".format(rr, cc)
                dest_straight = "top" + "{}_{}".format(rr, cc)
                dest_right = "right" + "{}_{}".format(rr + 1, cc)

                # Assumption: number of vertical_lanes = number of horizontal lanes

                return triple_cons(self.vertical_lanes, origin, dest_left, dest_straight, dest_right, signal_group)

            def build_right_conns(rr, cc):
                """Build the left, straight and right connections for the left horizontal edge
                entering the node. Returns a list of three dicts"""

                origin = "right" + "{}_{}".format(rr, cc)
                dest_left = "top" + "{}_{}".format(rr, cc)
                dest_straight = "right" + "{}_{}".format(rr + 1, cc)
                dest_right = "bot" + "{}_{}".format(rr, cc + 1)

                # Assumption: number of vertical_lanes = number of horizontal lanes

                return triple_cons(self.vertical_lanes, origin, dest_left, dest_straight, dest_right, signal_group)

            connections += build_bot_conns(row, col)
            connections += build_top_conns(row, col)
            connections += build_left_conns(row, col)
            connections += build_right_conns(row, col)

            return connections

        # build connections at each intersection node
        for x in range(1, x_max):
            for y in range(1, y_max):

                # node_id = "{}_{}".format(i, j)
                # right_node_id = "{}_{}".format(i, j + 1)
                # top_node_id = "{}_{}".format(i + 1, j)
                #
                # conn = []
                # for lane in range(self.vertical_lanes):
                #     conn += new_con("bot", node_id, right_node_id, lane, 1)
                #     conn += new_con("top", right_node_id, node_id, lane, 1)
                # for lane in range(self.horizontal_lanes):
                #     conn += new_con("right", node_id, top_node_id, lane, 2)
                #     conn += new_con("left", top_node_id, node_id, lane, 2)

                node_id = "({}.{})".format(x, y)
                con_dict[node_id] = node_cons(x, y, 1)

        return con_dict

    # TODO necessary?
    def specify_edge_starts(self):
        """See parent class."""
        edgestarts = []
        for i in range(self.col_num + 1):
            for j in range(self.row_num + 1):
                index = "{}_{}".format(j, i)
                if i != self.col_num:
                    edgestarts += [("left" + index, 0 + i * 50 + j * 5000),
                                   ("right" + index, 10 + i * 50 + j * 5000)]
                if j != self.row_num:
                    edgestarts += [("top" + index, 15 + i * 50 + j * 5000),
                                   ("bot" + index, 20 + i * 50 + j * 5000)]

        return edgestarts

    # TODO necessary?
    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """See parent class."""
        grid_array = net_params.additional_params["grid_array"]
        row_num = grid_array["row_num"]
        col_num = grid_array["col_num"]
        cars_heading_left = grid_array["cars_left"]
        cars_heading_right = grid_array["cars_right"]
        cars_heading_top = grid_array["cars_top"]
        cars_heading_bot = grid_array["cars_bot"]

        start_pos = []

        x0 = 6  # position of the first car
        dx = 10  # distance between each car

        start_lanes = []
        for i in range(col_num):
            start_pos += [("right0_{}".format(i), x0 + k * dx)
                          for k in range(cars_heading_right)]
            start_pos += [("left{}_{}".format(row_num, i), x0 + k * dx)
                          for k in range(cars_heading_left)]
            horz_lanes = np.random.randint(low=0, high=net_params.additional_params["horizontal_lanes"],
                                           size=cars_heading_left + cars_heading_right).tolist()
            start_lanes += horz_lanes

        for i in range(row_num):
            start_pos += [("top{}_{}".format(i, col_num), x0 + k * dx)
                          for k in range(cars_heading_top)]
            start_pos += [("bot{}_0".format(i), x0 + k * dx)
                          for k in range(cars_heading_bot)]
            vert_lanes = np.random.randint(low=0, high=net_params.additional_params["vertical_lanes"],
                                           size=cars_heading_left + cars_heading_right).tolist()
            start_lanes += vert_lanes

        return start_pos, start_lanes

    @property
    def node_mapping(self):
        """Map nodes to edges.
        Returns a list of pairs (node, connected edges) of all inner nodes
        and for each of them, the 4 edges that leave this node.
        The nodes are listed in alphabetical order, and within that, edges are
        listed in order: [bot, right, top, left].
        """
        mapping = {}

        for row in range(self.row_num):
            for col in range(self.col_num):
                node_id = "center{}".format(row * self.col_num + col)

                top_edge_id = "left{}_{}".format(row + 1, col)
                bot_edge_id = "right{}_{}".format(row, col)
                right_edge_id = "top{}_{}".format(row, col + 1)
                left_edge_id = "bot{}_{}".format(row, col)

                mapping[node_id] = [left_edge_id, bot_edge_id,
                                    right_edge_id, top_edge_id]

        return sorted(mapping.items(), key=lambda x: x[0])

    # TODO: I'll need to change the tests to reflect the new mapping method names
    @property
    def node_mapping_outer(self):
        """Map outer nodes, specifically, the start nodes, to edges.
        Returns a list of pairs (node, connected edges) of all outer nodes
        and for each of them, the edge that starts from this node and the edge that ends at this node..
        The nodes are listed in alphabetical order, and within that, edges are
        listed in order: [bot, right, top, left].
        """
        mapping = {}
        for col in range(self.col_num):
            for node_pos in (["bot"]):
                node_in_id = "{}_col_short{}".format(node_pos, col)
                mapping[node_in_id] = "right{}_{}".format(0, col)

                node_out_id = "{}_col_long{}".format(node_pos, col)
                mapping[node_out_id] = "left{}_{}".format(0, col)

        for col in range(self.col_num):
            for node_pos in (["top"]):
                node_in_id = "{}_col_short{}".format(node_pos, col)
                mapping[node_in_id] = "left{}_{}".format(self.row_num, col)

                node_out_id = "{}_col_short{}".format(node_pos, col)
                mapping[node_out_id] = "right{}_{}".format(self.row_num, col)

        for row in range(self.row_num):
            for node_pos in (["left"]):
                node_in_id = "{}_row_short{}".format(node_pos, row)
                mapping[node_in_id] = "bot{}_{}".format(row, 0)

                node_out_id = "{}_row_long{}".format(node_pos, row)
                mapping[node_out_id] = "top{}_{}".format(row, 0)

        for row in range(self.row_num):
            for node_pos in (["right"]):
                node_in_id = "{}_row_short{}".format(node_pos, row)
                mapping[node_in_id] = "top{}_{}".format(row, self.col_num)

                node_out_id = "{}_row_long{}".format(node_pos, row)
                mapping[node_out_id] = "bot{}_{}".format(row, self.col_num)

        return mapping
