import numpy as np

from cistar.core.params import InitialConfig
from cistar.scenarios.base_scenario import Scenario


class Figure8Scenario(Scenario):
    def __init__(self, name, generator_class, vehicles, net_params, initial_config=InitialConfig()):
        """
        Initializes a figure 8 scenario. Required net_params: radius_ring, lanes,
        speed_limit, resolution. Required initial_config: positions.

        See Scenario.py for description of params.
        """
        self.ring_edgelen = net_params.additional_params["radius_ring"] * np.pi / 2.
        self.intersection_len = 2 * net_params.additional_params["radius_ring"]
        self.junction_len = 2.9 + 3.3 * net_params.additional_params["lanes"]
        self.inner_space_len = 0.28

        # instantiate "length" in net params
        net_params.additional_params["length"] = \
            6 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + 10 * self.inner_space_len

        if "radius_ring" not in net_params.additional_params:
            raise ValueError("radius of ring not supplied")
        self.radius_ring = net_params.additional_params["radius_ring"]

        self.length = net_params.additional_params["length"]
        print(self.length)

        if "lanes" not in net_params.additional_params:
            raise ValueError("number of lanes not supplied")
        self.lanes = net_params.additional_params["lanes"]

        if "speed_limit" not in net_params.additional_params:
            raise ValueError("speed limit not supplied")
        self.speed_limit = net_params.additional_params["speed_limit"]

        if "resolution" not in net_params.additional_params:
            raise ValueError("resolution of circular sections not supplied")
        self.resolution = net_params.additional_params["resolution"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config=initial_config)

    def specify_edge_starts(self):
        """
        See base class
        """
        edgestarts = \
            [("bottom_lower_ring", 0 + self.inner_space_len),
             ("right_lower_ring_in", self.ring_edgelen + 2 * self.inner_space_len),
             ("right_lower_ring_out",
              self.ring_edgelen + self.intersection_len / 2 + self.junction_len + 3 * self.inner_space_len),
             ("left_upper_ring",
              self.ring_edgelen + self.intersection_len + self.junction_len + 4 * self.inner_space_len),
             ("top_upper_ring",
              2 * self.ring_edgelen + self.intersection_len + self.junction_len + 5 * self.inner_space_len),
             ("right_upper_ring",
              3 * self.ring_edgelen + self.intersection_len + self.junction_len + 6 * self.inner_space_len),
             ("bottom_upper_ring_in",
              4 * self.ring_edgelen + self.intersection_len + self.junction_len + 7 * self.inner_space_len),
             ("bottom_upper_ring_out",
              4 * self.ring_edgelen + 3 / 2 * self.intersection_len + 2 * self.junction_len + 8 * self.inner_space_len),
             ("top_lower_ring",
              4 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + 9 * self.inner_space_len),
             ("left_lower_ring",
              5 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + 10 * self.inner_space_len)]

        return edgestarts

    def specify_intersection_edge_starts(self):
        """
        See base class
        """
        intersection_edgestarts = \
            [(":center_intersection_%s" % (1+self.lanes),
              self.ring_edgelen + self.intersection_len / 2 + 3 * self.inner_space_len),
             (":center_intersection_1",
              4 * self.ring_edgelen + 3 / 2 * self.intersection_len + self.junction_len + 8 * self.inner_space_len)]

        return intersection_edgestarts

    def specify_internal_edge_starts(self):
        """
        See base class
        """
        internal_edgestarts = \
            [(":bottom_lower_ring", 0),
             (":right_lower_ring_in", self.ring_edgelen + self.inner_space_len),
             (":right_lower_ring_out",
              self.ring_edgelen + self.intersection_len / 2 + self.junction_len + 2 * self.inner_space_len),
             (":left_upper_ring",
              self.ring_edgelen + self.intersection_len + self.junction_len + 3 * self.inner_space_len),
             (":top_upper_ring",
              2 * self.ring_edgelen + self.intersection_len + self.junction_len + 4 * self.inner_space_len),
             (":right_upper_ring",
              3 * self.ring_edgelen + self.intersection_len + self.junction_len + 5 * self.inner_space_len),
             (":bottom_upper_ring_in",
              4 * self.ring_edgelen + self.intersection_len + self.junction_len + 6 * self.inner_space_len),
             (":bottom_upper_ring_out",
              4 * self.ring_edgelen + 3 / 2 * self.intersection_len + 2 * self.junction_len + 7 * self.inner_space_len),
             (":top_lower_ring",
              4 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + 8 * self.inner_space_len),
             (":left_lower_ring",
              5 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + 9 * self.inner_space_len)]

        return internal_edgestarts
