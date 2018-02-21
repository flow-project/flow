import math

from flow.scenarios.base_scenario import Scenario


# TODO(ak): make sure I go them all, and modify
ADDITIONAL_NET_PARAMS = {
    # (blank)
    "grid_array": {
        # (blank)
        "row_num": 2,
        # (blank)
        "col_num": 2,
        # (blank)
        "inner_length": None,
        # (blank)
        "short_length": None,
        # (blank)
        "long_length": None,
    },
    # specifies whether to add traffic lights to the intersections of the grid
    "traffic_lights": True,
    # number of lanes in the horizontal edges
    "horizontal_lanes": 1,
    # number of lanes in the vertical edges
    "vertical_lanes": 1
}


class SimpleGridScenario(Scenario):
    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=None):
        """Initializes an nxm grid scenario.

        Requires from net_params:
        - (blank): (blank)
        - (blank): (blank)
        - (blank): (blank)
        - (blank): (blank)

        In order for right-of-way dynamics to take place at the intersections,
        set "no_internal_links" in net_params to False.

        See Scenario.py for description of params.
        """
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

        net_params.additional_params["length"] = 10000  # TODO(ak): what is this for?

        # TODO(ak): we have this now. modify?
        # number of lanes in each edge
        self.lanes = {}
        for i in range(self.col_num-1):
            self.lanes.update(
                {"left{}_{}".format(self.row_num-1, i): vertical_lanes,
                 "right0_{}".format(i): horizontal_lanes})
        for i in range(self.row_num-1):
                self.lanes.update(
                    {"top{}_{}".format(i, self.col_num-1): vertical_lanes,
                     "bot" + str(i)+'_'+'0': horizontal_lanes})

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config=initial_config)

        self.edges = self.generator.specify_edges(net_params)

    # TODO, make this make any sense at all
    def specify_edge_starts(self):
        """Edges go in the following order: vert_right, vert_left, horz_right,
        horz_left"""
        vjl = self.vertical_junction_len
        hjl = self.horizontal_junction_len
        edgestarts = []
        curr_length = 0
        for i in range(self.col_num + 1):
            for j in range(self.row_num + 1):
                index = str(j) + '_' + str(i)

                edgestarts += \
                    [("left"+str(index), 0+i*50 + j*5000),
                     ("right"+str(index), 10+i*50 + j*5000),
                     ("top"+str(index), 15+i*50 + j*5000),
                     ("bot"+str(index), 20+i*50 + j*5000)
                    ]

        return edgestarts

    # TODO actually define the intersection edge starts
    def specify_intersection_edge_starts(self):  # used for get distance to intersections
        num = self.col_num - 1
        intersection_edgestarts = \
            [(":center", 0)]
        return intersection_edgestarts

    def gen_even_start_pos(self, initial_config, num_vehicles, **kwargs):
        row_num = self.grid_array["row_num"]
        col_num = self.grid_array["col_num"]
        num_cars = self.vehicles.num_vehicles
        per_edge = int(num_cars/(2 * (row_num+col_num)))
        start_positions = []
        d_inc = 6
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

    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        """ HARDCODING DONE NEED FIX """
        row_num = self.grid_array["row_num"]
        col_num = self.grid_array["col_num"]
        num_left = self.grid_array["cars_left"]
        num_right = self.grid_array["cars_right"]
        num_bot = self.grid_array["cars_bot"]
        num_top = self.grid_array["cars_top"]
        if "rl_veh" in self.grid_array:
            rl_veh = self.grid_array["rl_veh"]
        else: 
            rl_veh = 0

        start_positions = []
        d_inc = 8
        for i in range(self.col_num):
            x = 6
            for k in range(num_right):
                start_positions.append(("right0_{}".format(i), x))
                x += d_inc

            x = 6
            for k in range(num_left):
                start_positions.append(("left{}_{}".format(row_num, i), x))
                x += d_inc

        for i in range(self.row_num):
            x = 6
            for k in range(num_bot - math.ceil(rl_veh/2)):
                start_positions.append(("bot{}_0".format(i), x))
                x += d_inc
            start_positions.append(("rlbot{}_0".format(i), x))

            x = 6
            for k in range(num_top - (rl_veh//2)):
                start_positions.append(("top{}_{}".format(i, col_num), x))
                x += d_inc
            # Append any RL vehicles at the front
            start_positions.append(("rltop{}_{}".format(i, col_num), x))

        start_lanes = [0] * len(start_positions)
        return start_positions, start_lanes

    def get_edge_names(self):
        """Given a list of edge objects, returns a list of the edges' id
        attribute."""
        return [edge['id'] for edge in self.edges]

    def get_node_mapping(self):
        """Return a list of a dictionary of nodes mapped to a list of edges
        that head toward the node."""
        return sorted(self.generator.node_mapping.items(), key=lambda k: k[1])
