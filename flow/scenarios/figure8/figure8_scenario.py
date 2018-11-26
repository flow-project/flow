"""Contains the figure eight scenario class."""

import numpy as np

from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.scenarios.base_scenario import Scenario

ADDITIONAL_NET_PARAMS = {
    # radius of the circular components
    "radius_ring": 30,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # resolution of the curved portions
    "resolution": 40
}


class Figure8Scenario(Scenario):
    """Figure eight scenario class."""

    def __init__(self,
                 name,
                 generator_class,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initialize a figure 8 scenario.

        Requires from net_params:
        - ring_radius: radius of the circular portions of the network. Also
          corresponds to half the length of the perpendicular straight lanes.
        - resolution: number of nodes resolution in the circular portions
        - lanes: number of lanes in the network
        - speed: max speed of vehicles in the network

        In order for right-of-way dynamics to take place at the intersection,
        set "no_internal_links" in net_params to False.

        See flow/scenarios/base_scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        ring_radius = net_params.additional_params["radius_ring"]
        self.ring_edgelen = ring_radius * np.pi / 2.
        self.intersection_len = 2 * ring_radius
        self.junction_len = 8.0 + 3.2 * net_params.additional_params["lanes"]
        self.inner_space_len = 0.28

        # instantiate "length" attribute
        self.length = \
            6 * self.ring_edgelen + 2 * self.intersection_len + \
            2 * self.junction_len + 10 * self.inner_space_len
        self.lanes = net_params.additional_params["lanes"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """See base class."""
        edgestarts = \
            [("bottom_lower_ring",
              0 + self.inner_space_len),
             ("right_lower_ring_in",
              self.ring_edgelen + 2 * self.inner_space_len),
             ("right_lower_ring_out",
              self.ring_edgelen + self.intersection_len / 2 +
              self.junction_len + 3 * self.inner_space_len),
             ("left_upper_ring",
              self.ring_edgelen + self.intersection_len +
              self.junction_len + 4 * self.inner_space_len),
             ("top_upper_ring",
              2 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 5 * self.inner_space_len),
             ("right_upper_ring",
              3 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 6 * self.inner_space_len),
             ("bottom_upper_ring_in",
              4 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 7 * self.inner_space_len),
             ("bottom_upper_ring_out",
              4 * self.ring_edgelen + 3 / 2 * self.intersection_len +
              2 * self.junction_len + 8 * self.inner_space_len),
             ("top_lower_ring",
              4 * self.ring_edgelen + 2 * self.intersection_len +
              2 * self.junction_len + 9 * self.inner_space_len),
             ("left_lower_ring",
              5 * self.ring_edgelen + 2 * self.intersection_len +
              2 * self.junction_len + 10 * self.inner_space_len)]

        return edgestarts

    def specify_intersection_edge_starts(self):
        """See base class."""
        intersection_edgestarts = \
            [(":center_intersection_%s" % self.lanes,
              self.ring_edgelen + self.intersection_len / 2 +
              3 * self.inner_space_len),
             (":center_intersection_0",
              4 * self.ring_edgelen + 3 / 2 * self.intersection_len +
              self.junction_len + 8 * self.inner_space_len)]

        return intersection_edgestarts

    def specify_internal_edge_starts(self):
        """See base class."""
        internal_edgestarts = \
            [(":bottom_lower_ring",
              0),
             (":right_lower_ring_in",
              self.ring_edgelen + self.inner_space_len),
             (":right_lower_ring_out",
              self.ring_edgelen + self.intersection_len / 2 +
              self.junction_len + 2 * self.inner_space_len),
             (":left_upper_ring",
              self.ring_edgelen + self.intersection_len +
              self.junction_len + 3 * self.inner_space_len),
             (":top_upper_ring",
              2 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 4 * self.inner_space_len),
             (":right_upper_ring",
              3 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 5 * self.inner_space_len),
             (":bottom_upper_ring_in",
              4 * self.ring_edgelen + self.intersection_len +
              self.junction_len + 6 * self.inner_space_len),
             (":bottom_upper_ring_out",
              4 * self.ring_edgelen + 3 / 2 * self.intersection_len +
              2 * self.junction_len + 7 * self.inner_space_len),
             (":top_lower_ring",
              4 * self.ring_edgelen + 2 * self.intersection_len +
              2 * self.junction_len + 8 * self.inner_space_len),
             (":left_lower_ring",
              5 * self.ring_edgelen + 2 * self.intersection_len +
              2 * self.junction_len + 9 * self.inner_space_len)]

        return internal_edgestarts
