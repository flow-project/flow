import random

import numpy as np

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights


ADDITIONAL_NET_PARAMS = {
    # length of the horizontal edge before the intersection
    "horizontal_length_in": 400,
    # length of the horizontal edge after the intersection
    "horizontal_length_out": 10,
    # number of lanes in the horizontal edges
    "horizontal_lanes": 1,
    # length of the vertical edge before the intersection
    "vertical_length_in": 400,
    # length of the vertical edge after the intersection
    "vertical_length_out": 10,
    # number of lanes in the vertical edges
    "vertical_lanes": 1,
    # max speed limit of the vehicles on the road network
    "speed_limit": {"horizontal": 30, "vertical": 30}
}


class TwoWayIntersectionScenario(Scenario):

    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initializes a two-way intersection scenario.

        Requires from net_params:
        - horizontal_length_in: length of the horizontal edge before the
          intersection
        - horizontal_length_out: length of the horizontal edge after the
          intersection
        - horizontal_lanes: number of lanes in the horizontal edges
        - vertical_length_in: length of the vertical edge before the
          intersection
        - vertical_length_out: length of the vertical edge after the
          intersection
        - vertical_lanes: number of lanes in the vertical edges
        - speed_limit: max speed limit of the vehicles on the road network.
          May be a single value (for both lanes) or a dict separating the two,
          of the form: speed_limit = {"horizontal":{float}, "vertical":{float}}

        Required initial_config: positions.

        See Scenario.py for description of params.

        Note:
        -----
            Set no_internal_links in net_params to False to receive queueing
            at intersections.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.left_len = net_params.additional_params["horizontal_length_in"]
        self.right_len = net_params.additional_params["horizontal_length_out"]
        self.bottom_len = net_params.additional_params["vertical_length_in"]
        self.top_len = net_params.additional_params["vertical_length_out"]

        self.horizontal_junction_len = \
            2.9 + 3.3 * net_params.additional_params["vertical_lanes"]
        self.vertical_junction_len = \
            2.9 + 3.3 * net_params.additional_params["horizontal_lanes"]
        self.inner_space_len = 0.28

        # instantiate "length" in net params
        net_params.additional_params["length"] = self.left_len + self.right_len\
            + self.horizontal_junction_len + self.bottom_len + self.top_len\
            + self.vertical_junction_len

        self.lanes = {"top": net_params.additional_params["vertical_lanes"],
                      "bottom": net_params.additional_params["vertical_lanes"],
                      "left": net_params.additional_params["horizontal_lanes"],
                      "right": net_params.additional_params["horizontal_lanes"]}

        # enter_lane specifies which lane a car enters given a certain direction
        self.enter_lane = {"horizontal": "left", "vertical": "bottom"}

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        edgestarts = \
            [("bottom", 0),
             ("top", self.bottom_len + self.vertical_junction_len),
             ("left", self.bottom_len + self.vertical_junction_len
              + self.top_len),
             ("right", self.bottom_len + self.vertical_junction_len
              + self.top_len + self.left_len + self.horizontal_junction_len)]
        return edgestarts

    def specify_intersection_edge_starts(self):
        intersection_edgestarts = \
            [(":center_%s" % (1+self.lanes["left"]), self.bottom_len),
             (":center_1", (self.bottom_len + self.vertical_junction_len +
                            self.top_len) + self.left_len)]
        return intersection_edgestarts

    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
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
