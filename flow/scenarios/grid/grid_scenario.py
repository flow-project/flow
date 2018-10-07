"""Contains the grid scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights

ADDITIONAL_NET_PARAMS = {
    # dictionary of grid array data
    "grid_array": {
        # number of horizontal rows of edges
        "row_num": 3,
        # number of vertical columns of edges
        "col_num": 2,
        # length of inner edges in the grid network
        "inner_length": None,
        # length of edges that vehicles start on
        "short_length": None,
        # length of final edge in route
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
        "vertical": 35,
        "horizontal": 35
    },
}


class SimpleGridScenario(Scenario):
    """Grid scenario class."""

    def __init__(self,
                 name,
                 generator_class,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initialize an nxm grid scenario.

        The grid scenario consists of m vertical lanes and n horizontal lanes,
        with a total of nxm intersections where the vertical and horizontal
        edges meet.

        Requires from net_params:
        - grid_array: dictionary of grid array data, with the following keys
          - row_num: number of horizontal rows of edges
          - col_num: number of vertical columns of edges
          - inner_length: length of inner edges in the grid network
          - short_length: length of edges that vehicles start on
          - long_length: length of final edge in route
          - cars_top: number of cars starting at the edges heading to the top
          - cars_bot: number of cars starting at the edges heading to the
            bottom
          - cars_left: number of cars starting at the edges heading to the left
          - cars_right: number of cars starting at the edges heading to the
            right
        - horizontal_lanes: number of lanes in the horizontal edges
        - vertical_lanes: number of lanes in the vertical edges
        - speed_limit: speed limit for all edges. This may be represented as a
          float value, or a dictionary with separate values for vertical and
          horizontal lanes.

        In order for right-of-way dynamics to take place at the intersections,
        set "no_internal_links" in net_params to False.

        See flow/scenarios/base_scenario.py for description of params.
        """
        optional = ["tl_logic"]
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params and p not in optional:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        for p in ADDITIONAL_NET_PARAMS["grid_array"].keys():
            if p not in net_params.additional_params["grid_array"]:
                raise KeyError(
                    'Grid array parameter "{}" not supplied'.format(p))

        # this is a (mx1)x(nx1)x2 array
        # the third dimension is vertical length, horizontal length
        self.grid_array = net_params.additional_params["grid_array"]

        vertical_lanes = net_params.additional_params["vertical_lanes"]
        horizontal_lanes = net_params.additional_params["horizontal_lanes"]

        self.horizontal_junction_len = 2.9 + 3.3 * vertical_lanes
        self.vertical_junction_len = 2.9 + 3.3 * horizontal_lanes
        self.row_num = self.grid_array["row_num"]
        self.col_num = self.grid_array["col_num"]
        self.num_edges = (self.col_num+1) * self.row_num * 2 \
            + (self.row_num+1) * self.col_num * 2 + self.row_num * self.col_num
        self.inner_length = self.grid_array["inner_length"]
        self.short_length = self.grid_array["short_length"]
        self.long_length = self.grid_array["long_length"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    # TODO, make this make any sense at all
    def specify_edge_starts(self):
        """See parent class.

        Edges go in the following order: vert_right, vert_left, horz_right,
        horz_left.
        """
        edgestarts = []
        for i in range(self.col_num + 1):
            for j in range(self.row_num + 1):
                index = str(j) + '_' + str(i)
                edgestarts += [("left" + index, 0 + i * 50 + j * 5000),
                               ("right" + index, 10 + i * 50 + j * 5000),
                               ("top" + index, 15 + i * 50 + j * 5000),
                               ("bot" + index, 20 + i * 50 + j * 5000)]

        return edgestarts

    # TODO actually define the intersection edge starts
    # used for get distance to intersections
    def specify_intersection_edge_starts(self):
        """See parent class."""
        intersection_edgestarts = \
            [(":center", 0)]
        return intersection_edgestarts

    def gen_even_start_pos(self, initial_config, num_vehicles, **kwargs):
        """See parent class."""
        row_num = self.grid_array["row_num"]
        col_num = self.grid_array["col_num"]
        per_edge = int(num_vehicles / (2 * (row_num + col_num)))
        start_positions = []
        d_inc = 10
        for i in range(self.col_num):
            x = 6
            for k in range(per_edge):
                start_positions.append(("right0_{}".format(i), x))
                start_positions.append(("left{}_{}".format(row_num, i), x))
                x += d_inc

        for i in range(self.row_num):
            x = 6
            for k in range(per_edge):
                start_positions.append(("bot{}_0".format(i), x))
                start_positions.append(("top{}_{}".format(i, col_num), x))
                x += d_inc

        start_lanes = [0] * len(start_positions)
        start_speeds = [0] * len(start_positions)

        return start_positions, start_lanes, start_speeds

    def get_edge_names(self):
        """Return a the edge IDs attribute for a list of edge objects."""
        return [edge['id'] for edge in self.edges]

    def get_node_mapping(self):
        """Map nodes to edges.

        Returns a list of a dictionary of nodes mapped to a list of edges
        that head toward the node. Nodes are listed in alphabetical order
        and within that, edges are listed in order: [bot, right, top, left].
        """
        return sorted(self.generator.node_mapping.items(), key=lambda k: k[1])
