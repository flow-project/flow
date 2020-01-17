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
                 traffic_lights=TrafficLightParams(),
                 pedestrians=None):
        """Initialize an n*m traffic light grid network."""
        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights, pedestrians)
        self.use_traffic_lights = False
        self.nodes = self._nodes

    @property
    def _nodes(self):
        """See parent class"""
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
        car_1_start_pos = 20

        car_2_start_edge = "(1.2)--(1.1)"
        car_2_end_edge = "(1.1)--(2.1)"
        car_2_start_pos = 10

        car_3_start_edge = "(1.0)--(1.1)"
        car_3_end_edge = "(1.1)--(2.1)"
        car_3_start_pos = 0

        start_pos = [(car_1_start_edge, car_1_start_pos), (car_2_start_edge, car_2_start_pos), (car_3_start_edge, car_3_start_pos)]
        # In SUMO, lanes are zero-indexed starting from the right-most lane
        start_lanes = [0, 0, 0]

        return start_pos, start_lanes
