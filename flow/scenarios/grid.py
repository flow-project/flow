"""Contains the grid scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
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
    """Grid scenario class.

    The grid scenario consists of m vertical lanes and n horizontal lanes,
    with a total of nxm intersections where the vertical and horizontal
    edges meet.

    Requires from net_params:

    * **grid_array** : dictionary of grid array data, with the following keys

      * **row_num** : number of horizontal rows of edges
      * **col_num** : number of vertical columns of edges
      * **inner_length** : length of inner edges in the grid network
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

    In order for right-of-way dynamics to take place at the intersections,
    set *no_internal_links* in net_params to False.

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.scenarios import SimpleGridScenario
    >>>
    >>> scenario = SimpleGridScenario(
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
    >>>         no_internal_links=False  # we want junctions
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize an n*m grid scenario."""
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

        self.vertical_lanes = net_params.additional_params["vertical_lanes"]
        self.horizontal_lanes = net_params.additional_params[
            "horizontal_lanes"]

        # radius of the inner nodes
        self.inner_nodes_radius = 2.9 + 3.3 * max(self.vertical_lanes,
                                                  self.horizontal_lanes)

        self.row_num = self.grid_array["row_num"]
        self.col_num = self.grid_array["col_num"]
        self.num_edges = (self.col_num+1) * self.row_num * 2 \
            + (self.row_num+1) * self.col_num * 2 + self.row_num * self.col_num
        self.inner_length = self.grid_array["inner_length"]

        # vehicles spawn on short edge and exit on long edge
        self.short_length = self.grid_array["short_length"]
        self.long_length = self.grid_array["long_length"]

        # this is a dictionary containing inner length, long outer length,
        # short outer length, and number of rows and columns
        self.grid_array = net_params.additional_params["grid_array"]

        # specifies whether or not there will be traffic lights at the
        # intersections (True by default)
        self.use_traffic_lights = net_params.additional_params.get(
            "traffic_lights", True)

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
            "numLanes": horizontal_lanes,
            "speed": speed_limit["horizontal"]
        }, {
            "id": "vertical",
            "numLanes": vertical_lanes,
            "speed": speed_limit["vertical"]
        }]

        return types

    # ===============================
    # ============ UTILS ============
    # ===============================

    def _build_inner_nodes(self):
        """Build out the inner nodes of the scenario.

        The inner nodes correspond to the intersections between the roads. They
        are numbered from bottom left, increasing first across the columns and
        then across the rows.

        For example, the nodes in a grid with 2 rows and 3 columns would be
        indexed as follows:

            |     |     |
        --- 3 --- 4 --- 5 ---
            |     |     |
        --- 0 --- 1 --- 2 ---
            |     |     |

        The id of a node is then "center{index}", for instance "center0" for
        node 0, "center1" for node 1 etc.

        Returns
        ------
        list <dict>
            List of inner nodes
        """
        node_type = "traffic_light" if self.use_traffic_lights else "priority"

        nodes = []
        for row in range(self.row_num):
            for col in range(self.col_num):
                nodes.append({
                    "id": "center{}".format(row * self.col_num + col),
                    "x": col * self.inner_length,
                    "y": row * self.inner_length,
                    "type": node_type,
                    "radius": self.inner_nodes_radius
                })

        return nodes

    def _build_outer_nodes(self):
        """Build out the outer nodes of the scenario.

        The outer nodes correspond to the extremities of the roads. There are
        two at each extremity, one where the vehicles enter the scenario
        (inflow) and one where the vehicles exit the scenario (outflow).

        Consider the following scenario with 2 rows and 3 columns, where the
        extremities are marked by 'x', the rows are labeled from 0 to 1 and the
        columns are labeled from 0 to 2:

                     x         x         x
                     |         |         |
                     |         |         |
        Row 1  x-----|---------|---------|-----x (*)
                     |         |         |
                     |         |         |
                     |         |         |
        Row 0  x-----|---------|---------|-----x
                     |         |         |
                     |         |         |
                     x         x         x

                  Column 0  Column 1  Column 2

        On row i, there are two nodes at the left extremity of the row, labeled
        "left_row_short{i}" and "left_row_long{i}", as well as two nodes at the
        right extremity labeled "right_row_short{i}" and "right_row_long{i}".

        On column j, there are two nodes at the bottom extremity of the column,
        labeled "bot_col_short{j}" and "bot_col_long{j}", as well as two nodes
        at the top extremity labeled "top_col_short{j}" and "top_col_long{j}".

        The "short" nodes correspond to where vehicles enter the network while
        the "long" nodes correspond to where vehicles exit the network.

        For example, at extremity (*):
        - the id of the input node is "right_row_short1"
        - the id of the output node is "right_row_long1"

        Returns
        ------
        list <dict>
            List of outer nodes
        """
        nodes = []

        def create_node(x, y, name, i):
            return [{"id": name + str(i), "x": x, "y": y, "type": "priority"}]

        # build nodes at the extremities of columns
        for i in range(self.col_num):
            x = i * self.inner_length
            y = (self.row_num - 1) * self.inner_length
            nodes += create_node(x, - self.short_length, "bot_col_short", i)
            nodes += create_node(x, - self.long_length, "bot_col_long", i)
            nodes += create_node(x, y + self.short_length, "top_col_short", i)
            nodes += create_node(x, y + self.long_length, "top_col_long", i)

        # build nodes at the extremity of rows
        for i in range(self.row_num):
            x = (self.col_num - 1) * self.inner_length
            y = i * self.inner_length
            nodes += create_node(- self.short_length, y, "left_row_short", i)
            nodes += create_node(- self.long_length, y, "left_row_long", i)
            nodes += create_node(x + self.short_length, y, "right_row_short", i)
            nodes += create_node(x + self.long_length, y, "right_row_long", i)

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
                    "priority": 78,
                    "from": "center" + str(node_index + 1),
                    "to": "center" + str(node_index),
                    "length": inner_length
                }, {
                    "id": "bot" + index,
                    "type": "horizontal",
                    "priority": 78,
                    "from": "center" + str(node_index),
                    "to": "center" + str(node_index + 1),
                    "length": inner_length
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
                    "priority": 78,
                    "from": "center" + str(node_index_bot),
                    "to": "center" + str(node_index_top),
                    "length": inner_length
                }, {
                    "id": "left" + index,
                    "type": "vertical",
                    "priority": 78,
                    "from": "center" + str(node_index_top),
                    "to": "center" + str(node_index_bot),
                    "length": inner_length
                }]

        return edges

    def specify_connections(self, net_params):
        """See parent class."""
        lanes_horizontal = net_params.additional_params["horizontal_lanes"]
        lanes_vertical = net_params.additional_params["vertical_lanes"]

        row_num = self.grid_array["row_num"]
        col_num = self.grid_array["col_num"]
        con_dict = {}

        # build connections
        for i in range(row_num):
            for j in range(col_num):
                conn = []
                node_index = i * col_num + j
                node_id = "center{}".format(node_index)
                index = "{}_{}".format(i, j)
                for l in range(lanes_vertical):
                    conn += [
                        {"from": "bot" + index,
                         "to": "bot" + "{}_{}".format(i, j + 1),
                         "fromLane": str(l),
                         "toLane": str(l),
                         "signal_group": 1}
                    ]
                    conn += [
                        {"from": "top" + "{}_{}".format(i, j + 1),
                         "to": "top" + index,
                         "fromLane": str(l),
                         "toLane": str(l),
                         "signal_group": 1}
                        ]
                for l_h in range(lanes_horizontal):
                    conn += [
                        {"from": "right" + index,
                         "to": "right" + "{}_{}".format(i + 1, j),
                         "fromLane": str(l_h),
                         "toLane": str(l_h),
                         "signal_group": 2}
                    ]
                    conn += [
                        {"from": "left" + "{}_{}".format(i + 1, j),
                         "to": "left" + index,
                         "fromLane": str(l_h),
                         "toLane": str(l_h),
                         "signal_group": 2}
                    ]
                con_dict[node_id] = conn

        return con_dict

    def _build_outer_edges(self):
        """Build the outer edges.

        Starts with the bottom edges, then the top edges, then the left edges,
        then the right.

        Returns
        -------
        list of dict
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
                "priority": 78,
                "from": "bot_col_short" + str(i),
                "to": "center" + str(i),
                "length": short_length
            }, {
                "id": "left" + index,
                "type": "vertical",
                "priority": 78,
                "from": "center" + str(i),
                "to": "bot_col_long" + str(i),
                "length": long_length
            }]
            # top edges
            index = str(row_num) + '_' + str(i)
            center_start = (row_num - 1) * col_num
            self.node_mapping["center" + str(center_start + i)].append("left" +
                                                                       index)
            edges += [{
                "id": "left" + index,
                "type": "vertical",
                "priority": 78,
                "from": "top_col_short" + str(i),
                "to": "center" + str(center_start + i),
                "length": short_length
            }, {
                "id": "right" + index,
                "type": "vertical",
                "priority": 78,
                "from": "center" + str(center_start + i),
                "to": "top_col_long" + str(i),
                "length": long_length
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
                "priority": 78,
                "from": "left_row_short" + str(j),
                "to": "center" + str(j * col_num),
                "length": short_length
            }, {
                "id": "top" + index,
                "type": "horizontal",
                "priority": 78,
                "from": "center" + str(j * col_num),
                "to": "left_row_long" + str(j),
                "length": long_length
            }]
            # right edges
            index = str(j) + '_' + str(col_num)
            center_index = (j * col_num) + col_num - 1
            self.node_mapping["center" + str(center_index)].append("top" +
                                                                   index)
            edges += [{
                "id": "top" + index,
                "type": "horizontal",
                "priority": 78,
                "from": "right_row_short" + str(j),
                "to": "center" + str(center_index),
                "length": short_length
            }, {
                "id": "bot" + index,
                "type": "horizontal",
                "priority": 78,
                "from": "center" + str(center_index),
                "to": "right_row_long" + str(j),
                "length": long_length
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

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """See parent class."""
        grid_array = net_params.additional_params["grid_array"]
        row_num = grid_array["row_num"]
        col_num = grid_array["col_num"]
        cars_left = grid_array["cars_left"]
        cars_right = grid_array["cars_right"]
        cars_top = grid_array["cars_top"]
        cars_bot = grid_array["cars_bot"]

        start_positions = []
        d_inc = 10
        for i in range(col_num):
            x = 6
            for k in range(cars_right):
                start_positions.append(("right0_{}".format(i), x))
                x += d_inc
            x = 6
            for k in range(cars_left):
                start_positions.append(("left{}_{}".format(row_num, i), x))
                x += d_inc

        for i in range(row_num):
            x = 6
            for k in range(cars_top):
                start_positions.append(("top{}_{}".format(i, col_num), x))
                x += d_inc
            x = 6
            for k in range(cars_bot):
                start_positions.append(("bot{}_0".format(i), x))
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
