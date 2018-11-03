"""Contains the figure eight scenario class."""

import numpy as np
from numpy import pi, sin, cos, linspace

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
        self.junction_len = 2.9 + 3.3 * net_params.additional_params["lanes"]
        self.inner_space_len = 0.28

        # instantiate "length" in net params
        net_params.additional_params["length"] = \
            6 * self.ring_edgelen + 2 * self.intersection_len + \
            2 * self.junction_len + 10 * self.inner_space_len

        self.radius_ring = net_params.additional_params["radius_ring"]
        self.length = net_params.additional_params["length"]
        self.lanes = net_params.additional_params["lanes"]
        self.resolution = net_params.additional_params["resolution"]

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]

        nodes = [{
            "id": "center_intersection",
            "x": repr(0),
            "y": repr(0),
            "type": "priority"
        }, {
            "id": "top_upper_ring",
            "x": repr(r),
            "y": repr(2 * r),
            "type": "priority"
        }, {
            "id": "bottom_upper_ring_in",
            "x": repr(r),
            "y": repr(0),
            "type": "priority"
        }, {
            "id": "left_upper_ring",
            "x": repr(0),
            "y": repr(r),
            "type": "priority"
        }, {
            "id": "right_upper_ring",
            "x": repr(2 * r),
            "y": repr(r),
            "type": "priority"
        }, {
            "id": "top_lower_ring",
            "x": repr(-r),
            "y": repr(0),
            "type": "priority"
        }, {
            "id": "bottom_lower_ring",
            "x": repr(-r),
            "y": repr(-2 * r),
            "type": "priority"
        }, {
            "id": "left_lower_ring",
            "x": repr(-2 * r),
            "y": repr(-r),
            "type": "priority"
        }, {
            "id": "right_lower_ring_in",
            "x": repr(0),
            "y": repr(-r),
            "type": "priority"
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]
        resolution = net_params.additional_params["resolution"]
        ring_edgelen = r * pi / 2.
        intersection_edgelen = 2 * r

        # intersection edges
        edges = [{
            "id": "right_lower_ring_in",
            "type": "edgeType",
            "priority": "78",
            "from": "right_lower_ring_in",
            "to": "center_intersection",
            "length": repr(intersection_edgelen / 2)
        }, {
            "id": "right_lower_ring_out",
            "type": "edgeType",
            "priority": "78",
            "from": "center_intersection",
            "to": "left_upper_ring",
            "length": repr(intersection_edgelen / 2)
        }, {
            "id": "bottom_upper_ring_in",
            "type": "edgeType",
            "priority": "46",
            "from": "bottom_upper_ring_in",
            "to": "center_intersection",
            "length": repr(intersection_edgelen / 2)
        }, {
            "id": "bottom_upper_ring_out",
            "type": "edgeType",
            "priority": "46",
            "from": "center_intersection",
            "to": "top_lower_ring",
            "length": repr(intersection_edgelen / 2)
        }]

        # ring edges
        edges += [{
            "id":
            "left_upper_ring",
            "type":
            "edgeType",
            "from":
            "left_upper_ring",
            "to":
            "top_upper_ring",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * (1 - cos(t)), r * (1 + sin(t)))
                for t in linspace(0, pi / 2, resolution)
            ])
        }, {
            "id":
            "top_upper_ring",
            "type":
            "edgeType",
            "from":
            "top_upper_ring",
            "to":
            "right_upper_ring",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * (1 + sin(t)), r * (1 + cos(t)))
                for t in linspace(0, pi / 2, resolution)
            ])
        }, {
            "id":
            "right_upper_ring",
            "type":
            "edgeType",
            "from":
            "right_upper_ring",
            "to":
            "bottom_upper_ring_in",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * (1 + cos(t)), r * (1 - sin(t)))
                for t in linspace(0, pi / 2, resolution)
            ])
        }, {
            "id":
            "top_lower_ring",
            "type":
            "edgeType",
            "from":
            "top_lower_ring",
            "to":
            "left_lower_ring",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (-r + r * cos(t), -r + r * sin(t))
                for t in linspace(pi / 2, pi, resolution)
            ])
        }, {
            "id":
            "left_lower_ring",
            "type":
            "edgeType",
            "from":
            "left_lower_ring",
            "to":
            "bottom_lower_ring",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (-r + r * cos(t), -r + r * sin(t))
                for t in linspace(pi, 3 * pi / 2, resolution)
            ])
        }, {
            "id":
            "bottom_lower_ring",
            "type":
            "edgeType",
            "from":
            "bottom_lower_ring",
            "to":
            "right_lower_ring_in",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (-r + r * cos(t), -r + r * sin(t))
                for t in linspace(-pi / 2, 0, resolution)
            ])
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        types = [{
            "id": "edgeType",
            "numLanes": repr(lanes),
            "speed": repr(speed_limit)
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "bottom_lower_ring": [
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring"
            ],
            "right_lower_ring_in": [
                "right_lower_ring_in", "right_lower_ring_out",
                "left_upper_ring", "top_upper_ring", "right_upper_ring",
                "bottom_upper_ring_in", "bottom_upper_ring_out",
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring"
            ],
            "right_lower_ring_out": [
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in"
            ],
            "left_upper_ring": [
                "left_upper_ring", "top_upper_ring", "right_upper_ring",
                "bottom_upper_ring_in", "bottom_upper_ring_out",
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring",
                "right_lower_ring_in", "right_lower_ring_out"
            ],
            "top_upper_ring": [
                "top_upper_ring", "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring"
            ],
            "right_upper_ring": [
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring"
            ],
            "bottom_upper_ring_in": [
                "bottom_upper_ring_in", "bottom_upper_ring_out",
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring",
                "right_lower_ring_in", "right_lower_ring_out",
                "left_upper_ring", "top_upper_ring", "right_upper_ring"
            ],
            "bottom_upper_ring_out": [
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in"
            ],
            "top_lower_ring": [
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring",
                "right_lower_ring_in", "right_lower_ring_out",
                "left_upper_ring", "top_upper_ring", "right_upper_ring",
                "bottom_upper_ring_in", "bottom_upper_ring_out"
            ],
            "left_lower_ring": [
                "left_lower_ring", "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring"
            ]
        }

        return rts

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
            [(":center_intersection_%s" % (1 + self.lanes),
              self.ring_edgelen + self.intersection_len / 2 +
              3 * self.inner_space_len),
             (":center_intersection_1",
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
