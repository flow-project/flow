"""Contains the grid scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from collections import defaultdict

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

        # this is a dictionary containing inner length, long outer length,
        # short outer length, and number of rows and columns
        self.grid_array = net_params.additional_params["grid_array"]

        self.node_mapping = defaultdict(list)
        self.name = "BobLoblawsLawBlog"  # DO NOT CHANGE

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        nodes = []
        nodes += self._build_inner_nodes()
        nodes += self._build_outer_nodes()
        return nodes

    def specify_tll(self, net_params):
        """See parent class."""
        return self._build_inner_nodes()

    def specify_edges(self, net_params):
        """See parent class."""
        edges = []
        edges += self._build_inner_edges()
        edges += self._build_outer_edges()
        # Sort node_mapping in counterclockwise order
        self._order_nodes()
        return edges

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {}
        row_num = self.grid_array["row_num"]
        col_num = self.grid_array["col_num"]
        for i in range(row_num):
            route_arr_bot = []
            route_arr_top = []
            for j in range(col_num + 1):
                route_arr_bot += ["bot" + str(i) + '_' + str(j)]
                route_arr_top += ["top" + str(i) + '_' + str(col_num - j)]
            rts.update({"bot" + str(i) + '_' + '0': route_arr_bot})
            rts.update({"top" + str(i) + '_' + str(col_num): route_arr_top})

        for i in range(col_num):
            route_arr_left = []
            route_arr_right = []
            for j in range(row_num + 1):
                route_arr_right += ["right" + str(j) + '_' + str(i)]
                route_arr_left += ["left" + str(row_num - j) + '_' + str(i)]
            rts.update({"left" + str(row_num) + '_' + str(i): route_arr_left})
            rts.update({"right" + '0' + '_' + str(i): route_arr_right})

        return rts

    def specify_types(self, net_params):
        """See parent class."""
        add_params = net_params.additional_params
        horizontal_lanes = add_params["horizontal_lanes"]
        vertical_lanes = add_params["vertical_lanes"]
        if isinstance(add_params["speed_limit"], int) or \
                isinstance(add_params["speed_limit"], float):
            speed_limit = {
                "horizontal": add_params["speed_limit"],
                "vertical": add_params["speed_limit"]
            }
        else:
            speed_limit = add_params["speed_limit"]

        types = [{
            "id": "horizontal",
            "numLanes": repr(horizontal_lanes),
            "speed": repr(speed_limit["horizontal"])
        }, {
            "id": "vertical",
            "numLanes": repr(vertical_lanes),
            "speed": repr(speed_limit["vertical"])
        }]

        return types

    # ===============================
    # ============ UTILS ============
    # ===============================

    def _build_inner_nodes(self):
        """Build out the inner nodes of the system.

        The nodes are numbered from bottom left and increasing first across the
        columns and then across the rows. For example, in a 3x3 grid, we will
        have four inner nodes with the bottom left being 0, the bottom right
        being 1, the top left being 2, the top right being 3. The coordinate of
        the bottom left inner node is (0, 0).

        Yields
        ------
        list <dict>
            List of inner nodes
        """
        tls = self.net_params.additional_params.get("traffic_lights", True)
        node_type = "traffic_light" if tls else "priority"
        row_num = self.grid_array["row_num"]
        col_num = self.grid_array["col_num"]
        inner_length = self.grid_array["inner_length"]
        nodes = []
        # sweep up across columns
        for i in range(row_num):
            # sweep across rows
            for j in range(col_num):
                index = i * col_num + j
                x_center = j * inner_length
                y_center = i * inner_length
                nodes.append({
                    "id": "center" + str(index),
                    "x": repr(x_center),
                    "y": repr(y_center),
                    "type": node_type
                })
        return nodes

    def _build_outer_nodes(self):
        """Build out the column nodes.

        There are two in each column below the bottom row, and two in each
        column above the top row. They are numbered with regards to the column
        they are in. The bottom are labeled "bot_col_short" and "bot_col_long".
        Top are named similarly. We then repeat the same process for the outer
        row nodes

        Yields
        ------
        list <dict>
            List of column, row nodes
        """
        col_num = self.grid_array["col_num"]
        row_num = self.grid_array["row_num"]
        inner_length = self.grid_array["inner_length"]
        short_length = self.grid_array["short_length"]
        long_length = self.grid_array["long_length"]
        nodes = []
        for i in range(col_num):
            # build the bottom nodes
            nodes += [{
                "id": "bot_col_short" + str(i),
                "x": repr(i * inner_length),
                "y": repr(-short_length),
                "type": "priority"
            }, {
                "id": "bot_col_long" + str(i),
                "x": repr(i * inner_length),
                "y": repr(-long_length),
                "type": "priority"
            }]
            # build the top nodes
            nodes += [{
                "id": "top_col_short" + str(i),
                "x": repr(i * inner_length),
                "y": repr((row_num - 1) * inner_length + short_length),
                "type": "priority"
            }, {
                "id": "top_col_long" + str(i),
                "x": repr(i * inner_length),
                "y": repr((row_num - 1) * inner_length + long_length),
                "type": "priority"
            }]
        for i in range(row_num):
            # build the left nodes
            nodes += [{
                "id": "left_row_short" + str(i),
                "x": repr(-short_length),
                "y": repr(i * inner_length),
                "type": "priority"
            }, {
                "id": "left_row_long" + str(i),
                "x": repr(-long_length),
                "y": repr(i * inner_length),
                "type": "priority"
            }]
            # build the right nodes
            nodes += [{
                "id": "right_row_short" + str(i),
                "x": repr((col_num - 1) * inner_length + short_length),
                "y": repr(i * inner_length),
                "type": "priority"
            }, {
                "id": "right_row_long" + str(i),
                "x": repr((col_num - 1) * inner_length + long_length),
                "y": repr(i * inner_length),
                "type": "priority"
            }]
        return nodes

    def _build_inner_edges(self):
        """Build the inner edges.

        First we build all of the column edges. For the upper edge, it would be
        called right_i_j or left_i_j where i is the row number and j is the
        column to the right of it.

        For the vertical edges the notation would be bot_i_j or top_i_j where
        i is the row above it, and j is the column number.

        INDEXED FROM ZERO.
        """
        row_num = self.grid_array["row_num"]
        col_num = self.grid_array["col_num"]
        inner_length = self.grid_array["inner_length"]
        edges = []

        # Build the horizontal edges
        for i in range(row_num):
            for j in range(col_num - 1):
                node_index = i * col_num + j
                index = "{}_{}".format(i, j + 1)
                self.node_mapping["center{}".format(node_index +
                                                    1)].append("bot" + index)
                self.node_mapping["center{}".format(node_index)].append("top" +
                                                                        index)

                edges += [{
                    "id": "top" + index,
                    "type": "horizontal",
                    "priority": "78",
                    "from": "center" + str(node_index + 1),
                    "to": "center" + str(node_index),
                    "length": repr(inner_length)
                }, {
                    "id": "bot" + index,
                    "type": "horizontal",
                    "priority": "78",
                    "from": "center" + str(node_index),
                    "to": "center" + str(node_index + 1),
                    "length": repr(inner_length)
                }]

        # Build the vertical edges
        for i in range(row_num - 1):
            for j in range(col_num):
                node_index_bot = i * col_num + j
                node_index_top = (i + 1) * col_num + j
                index = str(i + 1) + '_' + str(j)
                self.node_mapping["center{}".format(node_index_top)].append(
                    "right" + index)
                self.node_mapping["center{}".format(node_index_bot)].append(
                    "left" + index)

                edges += [{
                    "id": "right" + index,
                    "type": "vertical",
                    "priority": "78",
                    "from": "center" + str(node_index_bot),
                    "to": "center" + str(node_index_top),
                    "length": repr(inner_length)
                }, {
                    "id": "left" + index,
                    "type": "vertical",
                    "priority": "78",
                    "from": "center" + str(node_index_top),
                    "to": "center" + str(node_index_bot),
                    "length": repr(inner_length)
                }]

        return edges

    def _build_outer_edges(self):
        """Build the outer edges.

        Starts with the bottom edges, then the top edges, then the left edges,
        then the right.

        Yields
        ------
        list <dict>
            List of outer edges
        """
        row_num = self.grid_array["row_num"]
        col_num = self.grid_array["col_num"]
        short_length = self.grid_array["short_length"]
        long_length = self.grid_array["long_length"]
        edges = []

        # create dictionary of node to edges that go to it
        for i in range(col_num):
            index = '0_' + str(i)
            # bottom edges
            self.node_mapping["center" + str(i)].append("right" + index)
            edges += [{
                "id": "right" + index,
                "type": "vertical",
                "priority": "78",
                "from": "bot_col_short" + str(i),
                "to": "center" + str(i),
                "length": repr(short_length)
            }, {
                "id": "left" + index,
                "type": "vertical",
                "priority": "78",
                "from": "center" + str(i),
                "to": "bot_col_long" + str(i),
                "length": repr(long_length)
            }]
            # top edges
            index = str(row_num) + '_' + str(i)
            center_start = (row_num - 1) * col_num
            self.node_mapping["center" + str(center_start + i)].append("left" +
                                                                       index)
            edges += [{
                "id": "left" + index,
                "type": "vertical",
                "priority": "78",
                "from": "top_col_short" + str(i),
                "to": "center" + str(center_start + i),
                "length": repr(short_length)
            }, {
                "id": "right" + index,
                "type": "vertical",
                "priority": "78",
                "from": "center" + str(center_start + i),
                "to": "top_col_long" + str(i),
                "length": repr(long_length)
            }]

        # build the left and then the right edges
        for j in range(row_num):
            index = str(j) + '_0'
            # left edges
            self.node_mapping["center" + str(j * col_num)].append("bot" +
                                                                  index)
            edges += [{
                "id": "bot" + index,
                "type": "horizontal",
                "priority": "78",
                "from": "left_row_short" + str(j),
                "to": "center" + str(j * col_num),
                "length": repr(short_length)
            }, {
                "id": "top" + index,
                "type": "horizontal",
                "priority": "78",
                "from": "center" + str(j * col_num),
                "to": "left_row_long" + str(j),
                "length": repr(long_length)
            }]
            # right edges
            index = str(j) + '_' + str(col_num)
            center_index = (j * col_num) + col_num - 1
            self.node_mapping["center" + str(center_index)].append("top" +
                                                                   index)
            edges += [{
                "id": "top" + index,
                "type": "horizontal",
                "priority": "78",
                "from": "right_row_short" + str(j),
                "to": "center" + str(center_index),
                "length": repr(short_length)
            }, {
                "id": "bot" + index,
                "type": "horizontal",
                "priority": "78",
                "from": "center" + str(center_index),
                "to": "right_row_long" + str(j),
                "length": repr(long_length)
            }]

        return edges

    def _order_nodes(self):
        for node in self.node_mapping:
            adj_edges = ["" for _ in range(4)]
            for e in self.node_mapping[node]:
                if 'bot' in e:
                    adj_edges[0] = e
                elif 'right' in e:
                    adj_edges[1] = e
                elif 'top' in e:
                    adj_edges[2] = e
                elif 'left' in e:
                    adj_edges[3] = e
            self.node_mapping[node] = adj_edges

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
        return start_positions, start_lanes

    def get_edge_names(self):
        """Return a the edge IDs attribute for a list of edge objects."""
        return [edge['id'] for edge in self.edges]

    def get_node_mapping(self):
        """Map nodes to edges.

        Returns a list of a dictionary of nodes mapped to a list of edges
        that head toward the node. Nodes are listed in alphabetical order
        and within that, edges are listed in order: [bot, right, top, left].
        """
        return sorted(self.node_mapping.items(), key=lambda k: k[1])
