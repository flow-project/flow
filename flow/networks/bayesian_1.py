"""Contains the bayesian scenario 1 class."""

from flow.networks.traffic_light_grid import TrafficLightGridNetwork
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
import numpy as np
import networkx as nx
from itertools import islice

ADDITIONAL_NET_PARAMS = {
    # dictionary of bayesian scenario 1 grid array data
    "grid_array": {
        # number of horizontal rows of edges
        "row_num": 1,
        # number of vertical columns of edges
        "col_num": 1,
        # length of edges in the traffic light grid network
        "inner_length": None,
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


class Bayesian1Network(TrafficLightGridNetwork):
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
    >>>                 'row_num': 1,
    >>>                 'col_num': 1,
    >>>                 'inner_length': 500,
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
        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)
        self.use_traffic_lights = False
        self.nodes = self._nodes

    @property
    def _nodes(self):
        """See parent class"""
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

    def specify_nodes(self, net_params):
        """See parent class."""
        return self._nodes

    def specify_edges(self, net_params):
        """See parent class."""
        return self._edges

    def specify_routes(self, net_params):

        car_1_start_edge = "(2.1)--(1.1)"
        car_1_end_edge = "(1.1)--(0.1)"

        car_2_start_edge = "(1.2)--(1.1)"
        car_2_end_edge = "(1.1)--(2.1)"

        car_3_start_edge = "(1.0)--(1.1)"
        car_3_end_edge = "(1.1)--(2.1)"

        rts = {car_1_start_edge: [car_1_start_edge, car_1_end_edge],
               car_2_start_edge: [car_2_start_edge, car_2_end_edge],
               car_3_start_edge: [car_3_start_edge, car_3_end_edge]}

        return rts

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

        node_type = "traffic_light" if self.use_traffic_lights else "allway_stop"
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

        node_type = "traffic_light" if self.use_traffic_lights else "priority"
        x_max = self.col_num + 1
        y_max = self.row_num + 1

        nodes = []
        for x in range(1, x_max):
            for y in range(1, y_max):
                nodes.append({
                    "id": "({}.{})".format(x, y),
                    "x": x * self.inner_length,
                    "y": y * self.inner_length,
                    "type": node_type,
                    "radius": self.nodes_radius
                })

        return nodes

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

    def specify_connections(self, net_params, legal_turns=True):
        """Build out connections at each node (aside from those nodes where vehicles enter and exit).
        Connections describe what happens at the intersections. We specify the connections of an entire network using
        a dict, where keys are the individual node ids and the values are a list of all connections directly related to
        a particular node. For a particular node, we specify the node's connections using a list of "from" edges and "to"
        edges. Here, we link lanes with all possible adjacent lanes. This means vehicles can make
        turns at any intersection and can turn to any particular adjacent lane.

        Movement restrictions:
        Right lanes can travel only straight or right.
        Middle lanes can travel only straight.
        Left lanes can travel only straight or left.

        (N.B. turns can only be legal turns i.e. allowed by law)

        Returns
        -------
        dict<list<dict>>
            Dict of all the connections in the network
        """
        con_dict = {}
        x_max = self.col_num + 1
        y_max = self.row_num + 1

        def node_cons(x, y, signal_group):
            """Takes the (x, y) co-ordinates of an intersection node, and returns a list of dicts. Each dict specifies
            a single connections from a specific lane to another specific lane. We build ALL possible connections
            for this intersection, apart from the connection from a lane back to a lane in the same edge of the opposite
            direction."""

            def single_con_dict(from_edge, to_edge, from_lane, to_lane, signal_group):
                """Takes in the information for one specific connection and
                returns the dict to specify the connection"""

                return [{
                    "from": from_edge,
                    "to": to_edge,
                    "fromLane": str(from_lane),
                    "toLane": str(to_lane),
                    "signal_group": signal_group
                }]

            node_cons_list = []
            # origin_edges = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, top, left, bottom

            # Ids of the nodes adjacent to the intersection node
            right_node_id = "({}.{})".format(x + 1, y)
            top_node_id = "({}.{})".format(x, y + 1)
            left_node_id = "({}.{})".format(x - 1, y)
            bottom_node_id = "({}.{})".format(x, y - 1)
            center_node_id = "({}.{})".format(x, y)

            # {}_edge_in denotes an edge coming into the specified intersection node
            # {}_edge_out denotes an edge leaving the specified intersection node

            right_edge_in = right_node_id + "--" + center_node_id
            top_edge_in = top_node_id + "--" + center_node_id
            left_edge_in = left_node_id + "--" + center_node_id
            bottom_edge_in = bottom_node_id + "--" + center_node_id

            right_edge_out = center_node_id + "--" + right_node_id
            top_edge_out = center_node_id + "--" + top_node_id
            left_edge_out = center_node_id + "--" + left_node_id
            bottom_edge_out = center_node_id + "--" + bottom_node_id

            """
            Movement restrictions:
            Right lanes can travel only straight or right.            
            Middle lanes can travel only straight.
            Left lanes can travel only straight or left.

            In SUMO, lanes are numbered from 0, starting from the rightmost lane. Thus, a legal left turn by a left lane
            would be lane n to lane n. A legal right turn from the rightmost lane would be lane 0 to lane 0."""

            right_most_lane, left_most_lane = 0, self.horizontal_lanes - 1

            # TODO: ONLY leftmost lanes can turn left i.e stop non leftmost lanes from turning left as well
            # build vertical connections for RIGHT edge (1,0)
            for hor_l in range(self.horizontal_lanes):
                for vert_l in range(self.vertical_lanes):
                    # TODO: fix the strange lane turns
                    if legal_turns:
                        if hor_l == vert_l:
                            if hor_l == right_most_lane and vert_l == right_most_lane:  # only right most lane can turn right
                                print(right_edge_in, top_edge_out, hor_l, vert_l)
                                node_cons_list += single_con_dict(right_edge_in, top_edge_out, hor_l, vert_l,
                                                                  signal_group)
                            if hor_l == left_most_lane and vert_l == left_most_lane:  # only left most lane can turn left
                                node_cons_list += single_con_dict(right_edge_in, bottom_edge_out, hor_l, vert_l,
                                                                  signal_group)
                    else:
                        if hor_l == right_most_lane and vert_l == right_most_lane:  # only right most lane can turn right
                            node_cons_list += single_con_dict(right_edge_in, top_edge_out, hor_l, vert_l, signal_group)
                        if hor_l == left_most_lane and vert_l == left_most_lane:  # only left most lane can turn left
                            node_cons_list += single_con_dict(right_edge_in, bottom_edge_out, hor_l, vert_l,
                                                              signal_group)

            # build horizontal connection for RIGHT edge (1,0)
            for hor_l1 in range(self.horizontal_lanes):
                for hor_l2 in range(self.horizontal_lanes):
                    if hor_l1 == hor_l2:  # when going straight, you can only go directly straight
                        node_cons_list += single_con_dict(right_edge_in, left_edge_out, hor_l1, hor_l2, signal_group)

            # build vertical connections for LEFT edge (-1,0)
            for hor_l in range(self.horizontal_lanes):
                for vert_l in range(self.vertical_lanes):
                    if legal_turns:
                        if hor_l == vert_l:  # only allow legal lane transitions
                            if hor_l == right_most_lane and vert_l == right_most_lane:  # only right most lane can turn right
                                node_cons_list += single_con_dict(left_edge_in, bottom_edge_out, hor_l, vert_l,
                                                                  signal_group)
                            if hor_l == left_most_lane and vert_l == left_most_lane:  # only left most lane can turn left
                                node_cons_list += single_con_dict(left_edge_in, top_edge_out, hor_l, vert_l,
                                                                  signal_group)
                    else:
                        if hor_l == right_most_lane and vert_l == right_most_lane:  # only right most lane can turn right
                            node_cons_list += single_con_dict(left_edge_in, top_edge_out, hor_l, vert_l, signal_group)
                        if hor_l == left_most_lane and vert_l == left_most_lane:  # only left most lane can turn left
                            node_cons_list += single_con_dict(left_edge_in, bottom_edge_out, hor_l, vert_l,
                                                              signal_group)

            # build horizontal connection for LEFT edge (-1,0)
            for hor_l1 in range(self.horizontal_lanes):
                for hor_l2 in range(self.horizontal_lanes):
                    if hor_l1 == hor_l2:
                        node_cons_list += single_con_dict(left_edge_in, right_edge_out, hor_l1, hor_l2, signal_group)

            # build vertical connection for TOP edge (0, 1)
            for vert_l1 in range(self.vertical_lanes):
                for vert_l2 in range(self.vertical_lanes):
                    if vert_l1 == vert_l2:
                        node_cons_list += single_con_dict(top_edge_in, bottom_edge_out, vert_l1, vert_l2, signal_group)

            # build horizontal connections for TOP edge (0, 1)
            for vert_l in range(self.vertical_lanes):
                for hor_l in range(self.horizontal_lanes):
                    if legal_turns:
                        if vert_l == hor_l:  # only allow legal lane transitions
                            if vert_l == right_most_lane and hor_l == right_most_lane:  # only right most lane can turn right
                                node_cons_list += single_con_dict(top_edge_in, left_edge_out, vert_l, hor_l,
                                                                  signal_group)
                            if vert_l == left_most_lane and hor_l == left_most_lane:  # only left most lane can turn left
                                node_cons_list += single_con_dict(top_edge_in, right_edge_out, vert_l, hor_l,
                                                                  signal_group)
                    else:
                        if hor_l == right_most_lane and vert_l == right_most_lane:  # only right most lane can turn right
                            node_cons_list += single_con_dict(top_edge_in, left_edge_out, vert_l, hor_l, signal_group)
                        if hor_l == left_most_lane and vert_l == left_most_lane:  # only left most lane can turn left
                            node_cons_list += single_con_dict(top_edge_in, right_edge_out, vert_l, hor_l, signal_group)

            # build vertical connection for BOTTOM edge (0, -1)
            for vert_l1 in range(self.horizontal_lanes):
                for vert_l2 in range(self.vertical_lanes):
                    if vert_l1 == vert_l2:
                        node_cons_list += single_con_dict(bottom_edge_in, top_edge_out, vert_l1, vert_l2, signal_group)

            # build horizontal connections for BOTTOM edge (0, -1)
            for hor_l in range(self.horizontal_lanes):
                for vert_l in range(self.vertical_lanes):
                    # if legal_turns:
                    #     if hor_l == vert_l:
                    #         node_cons_list += single_con_dict(bottom_edge_in, left_edge_out, vert_l, hor_l, signal_group)
                    #         node_cons_list += single_con_dict(bottom_edge_in, right_edge_out, vert_l, hor_l, signal_group)
                    # else:
                    #     node_cons_list += single_con_dict(bottom_edge_in, left_edge_out, vert_l, hor_l, signal_group)
                    #     node_cons_list += single_con_dict(bottom_edge_in, right_edge_out, vert_l, hor_l, signal_group)
                    if legal_turns:
                        if hor_l == vert_l:  # only allow legal lane transitions
                            if hor_l == right_most_lane and vert_l == right_most_lane:  # only right most lane can turn right
                                node_cons_list += single_con_dict(bottom_edge_in, right_edge_out, hor_l, vert_l,
                                                                  signal_group)
                            if hor_l == left_most_lane and vert_l == left_most_lane:  # only left most lane can turn left
                                node_cons_list += single_con_dict(bottom_edge_in, left_edge_out, hor_l, vert_l,
                                                                  signal_group)
                    else:
                        if hor_l == right_most_lane and vert_l == right_most_lane:  # only right most lane can turn right
                            node_cons_list += single_con_dict(bottom_edge_in, right_edge_out, hor_l, vert_l,
                                                              signal_group)
                        if hor_l == left_most_lane and vert_l == left_most_lane:  # only left most lane can turn left
                            node_cons_list += single_con_dict(bottom_edge_in, left_edge_out, hor_l, vert_l,
                                                              signal_group)

            return node_cons_list

        # build connections at each intersection node
        for x in range(1, x_max):
            for y in range(1, y_max):
                node_id = "({}.{})".format(x, y)
                con_dict[node_id] = node_cons(x, y, 1)  # Still confused about what a signal_group does...,
                # but made all connections with all lanes!

        return con_dict

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

    # TODO necessary? KevinLin Note that initial_config isn't used here at all
    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """See parent class for full explanation

        Return 2 lists:
        1. list of start positions [(edge0, pos0), (edge1, pos1), ...]
        2. list of start lanes [lane0, lane1, lane 2, ...]"""

        # pos = 0 starts from the starting node of the edge
        car_1_start_edge = "(2.1)--(1.1)"
        car_1_end_edge = "(1.1)--(0.1)"
        car_1_start_pos = 30

        car_2_start_edge = "(1.2)--(1.1)"
        car_2_end_edge = "(1.1)--(2.1)"
        car_2_start_pos = 20

        car_3_start_edge = "(1.0)--(1.1)"
        car_3_end_edge = "(1.1)--(2.1)"
        car_3_start_pos = 10

        start_pos = [(car_1_start_edge, car_1_start_pos), (car_2_start_edge, car_2_start_pos), (car_3_start_edge, car_3_start_pos)]
        # In SUMO, lanes are zero-indexed starting from the right-most lane
        start_lanes = [0, 0, 0]

        return start_pos, start_lanes
