import random

import numpy as np

from cistar.scenarios.intersections.gen import *
from cistar.scenarios.base_scenario import Scenario


class TwoWayIntersectionScenario(Scenario):

    def __init__(self, name, generator_class, vehicles, net_params, cfg_params=None,
                 initial_config=None):
        """
        Initializes a two-way intersection scenario. Required net_params: horizontal_length_before,
        horizontal_length_after, horizontal_lanes, vertical_length_before, vertical_length_after, vertical_lanes,
        speed_limit. Required initial_config: positions.

        See Scenario.py for description of params.
        """
        self.left_len = net_params["horizontal_length_in"]
        self.right_len = net_params["horizontal_length_out"]
        self.bottom_len = net_params["vertical_length_in"]
        self.top_len = net_params["vertical_length_out"]

        self.horizontal_junction_len = 2.9 + 3.3 * net_params["vertical_lanes"]
        self.vertical_junction_len = 2.9 + 3.3 * net_params["horizontal_lanes"]
        self.inner_space_len = 0.28

        # instantiate "length" in net params
        net_params["length"] = self.left_len + self.right_len + self.horizontal_junction_len + \
            self.bottom_len + self.top_len + self.vertical_junction_len

        if "horizontal_lanes" not in net_params:
            raise ValueError("number of horizontal lanes not supplied")

        if "vertical_lanes" not in net_params:
            raise ValueError("number of vertical lanes not supplied")

        self.lanes = {"top": net_params["vertical_lanes"], "bottom": net_params["vertical_lanes"],
                      "left": net_params["horizontal_lanes"], "right": net_params["horizontal_lanes"]}

        # enter_lane specifies which lane a car enters given a certain direction
        self.enter_lane = {"horizontal": "left", "vertical": "bottom"}

        if "speed_limit" not in net_params:
            raise ValueError("speed limit not supplied")

        # if the speed limit is a single number, then all lanes have the same speed limit
        if isinstance(net_params["speed_limit"], int) or isinstance(net_params["speed_limit"], float):
            self.speed_limit = {"horizontal": net_params["speed_limit"],
                                "vertical": net_params["speed_limit"]}
        # if the speed limit is a dict with separate values for vertical and horizontal,
        # then they are set as such
        elif "vertical" in net_params["speed_limit"] and "horizontal" in net_params["speed_limit"]:
            self.speed_limit = {"horizontal": net_params["speed_limit"]["horizontal"],
                                "vertical": net_params["speed_limit"]["vertical"]}
        else:
            raise ValueError('speed limit must contain a number or a dict with keys: "vertical" and "horizontal"')

        super().__init__(name, generator_class, vehicles, net_params, cfg_params=cfg_params,
                         initial_config=initial_config)

    def specify_edge_starts(self):
        edgestarts = \
            [("bottom", 0),
             ("top", self.bottom_len + self.vertical_junction_len),
             ("left", 1000 * (self.bottom_len + self.vertical_junction_len + self.top_len)),
             ("right", 1000 * (self.bottom_len + self.vertical_junction_len + self.top_len) +
              self.left_len + self.horizontal_junction_len)]
        return edgestarts

    def specify_intersection_edge_starts(self):
        intersection_edgestarts = \
            [(":center_%s" % (1+self.lanes["left"]), self.bottom_len),
             (":center_1", 1000 * (self.bottom_len + self.vertical_junction_len + self.top_len) + self.left_len)]
        return intersection_edgestarts

    def gen_custom_start_pos(self, initial_config, **kwargs):
        """
        Generate random positions starting from the ends of the track.
        Vehicles are spaced so that no car can arrive at the 
        control portion of the track more often than...
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]    
        """
        rate = initial_config.additional_params["intensity"]
        v_enter = initial_config.additional_params["enter_speed"]

        start_positions = []
        x = 1
        # Fix it so processes in both lanes are poisson with the right
        # intensity, rather than half the intensity
        while len(start_positions) < self.vehicles.num_vehicles:
            left_lane = np.random.randint(2, size=1)
            d_inc = v_enter*random.expovariate(1.0/rate)
            # FIXME to get length of car that has been placed already
            # This should be the car length, other values are to make problem
            # easier
            if d_inc > 10:
                x += d_inc
                if left_lane:
                    start_positions.append(("left", x))
                else:
                    start_positions.append(("bottom", x))

        start_lanes = [0] * len(start_positions)

        return start_positions, start_lanes
