"""Contains the intersection network class."""

import numpy as np
from numpy import linspace

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks import Network

ADDITIONAL_NET_PARAMS = {
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # length of the four edges
    "edge_length": 100,
    # one way or two way?
    "one_way": True
}

# Note: To add turns, edit the

class SimpleIntNetwork(Network):
    """Intersection network class."""

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize an intersection network.
        Requires from net_params:
        - lanes: number of lanes in the network
        - speed_limit: max speed of vehicles in the network
        - edge_length: length of the four edges
        In order for right-of-way dynamics to take place at the intersection,
        set "no_internal_links" in net_params to False.
        See flow/network/base.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.lanes = net_params.additional_params["lanes"]
        self.speed_limit = net_params.additional_params["lanes"]
        self.edge_length = net_params.additional_params["edge_length"]
        self.junction_radius = (2.9 + 3.3 * self.lanes) / 2
        self.turns = net_params.additional_params["turns_on"]
        self.one_way = net_params.additional_params["one_way"]

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""

        nodes = [{
            "id": "center",
            "x": 0,
            "y": 0,
            "radius": self.junction_radius,
            "type": "priority"
        }, {
            "id": "right",
            "x": self.edge_length,
            "y": 0,
            "type": "priority"
        }, {
            "id": "top",
            "x": 0,
            "y": self.edge_length,
            "type": "priority"
        }, {
            "id": "left",
            "x": -self.edge_length,
            "y": 0,
            "type": "priority"
        }, {
            "id": "bottom",
            "x": 0,
            "y": -self.edge_length,
            "type": "priority"
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        edges = [{
            "id": "bottom_center",
            "type": "edgeType",
            "priority": 78,
            "from": "bottom",
            "to": "center",
            "length": self.edge_length
        }, {
            "id": "center_top",
            "type": "edgeType",
            "priority": 78,
            "from": "center",
            "to": "top",
            "length": self.edge_length
        }, {
            "id": "center_right",
            "type": "edgeType",
            "priority": 46,
            "from": "center",
            "to": "right",
            "length": self.edge_length
        }, {
            "id": "left_center",
            "type": "edgeType",
            "priority": 46,
            "from": "left",
            "to": "center",
            "length": self.edge_length
        }]

        if not self.one_way:
            edges += [{
                "id": "center_bottom",
                "type": "edgeType",
                "priority": 78,
                "from": "center",
                "to": "bottom",
                "length": self.edge_length
            }, {
                "id": "top_center",
                "type": "edgeType",
                "priority": 78,
                "from": "top",
                "to": "center",
                "length": self.edge_length
            }, {
                "id": "right_center",
                "type": "edgeType",
                "priority": 46,
                "from": "right",
                "to": "center",
                "length": self.edge_length
            }, {
                "id": "center_left",
                "type": "edgeType",
                "priority": 46,
                "from": "center",
                "to": "left",
                "length": self.edge_length
            }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        types = [{
            "id": "edgeType",
            "numLanes": self.lanes,
            "speed": self.speed_limit
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        if self.turns:
            # Probability that cars coming from the bottom road will turn right onto right road
            prob_right = 0.5
            # Probability that cars coming from the left road will turn left onto top road
            prob_left = 0.5
            rts = {
                "bottom_center": [(["bottom_center", "center_top"], 1-prob_right),
                        (["bottom_center", "center_right"], prob_right)],
                "left_center": [(["left_center", "center_right"], 1-prob_left),
                        (["left_center", "center_top"], prob_left)]
            }
            if not self.one_way:
                # Specify special turning probs for two-way and add all 3 possible routes in
                prob_left = 0.33
                prob_right = 0.33
                prob_straight = 1 - prob_left + prob_right
                rts = {
                    "bottom_center": [(["bottom_center", "center_top"], prob_straight),
                            (["bottom_center", "center_right"], prob_right),
                            (["bottom_center", "center_left"], prob_left)],
                    "left_center": [(["left_center", "center_right"], prob_straight),
                            (["left_center", "center_top"], prob_left),
                            (["left_center", "center_bottom"], prob_right)],
                    "top_center": [(["top_center", "center_bottom"], prob_straight),
                                   (["top_center", "center_right"], prob_left),
                                   (["top_center", "center_left"], prob_right)],
                    "right_center": [(["right_center", "center_left"], prob_straight),
                                     (["right_center", "center_bottom"], prob_left),
                                     (["right_center", "center_top"], prob_right)]
                }
        else:
            # Do no turns
            rts = {
                "bottom_center":["bottom_center", "center_top"],
                "left_center": ["left_center", "center_right"]
            }
            if not self.one_way:
                rts["top_center"] = ["top_center", "center_bottom"]
                rts["right_center"] = ["right_center", "center_left"]
        return rts

    def specify_connections(self, net_params):
        """See parent class."""
        conn = []
        for i in range(self.lanes):
            conn += [{"from": "bottom_center",
                      "to": "center_top",
                      "fromLane": str(i),
                      "toLane": str(i)}]
            conn += [{"from": "left_center",
                      "to": "center_right",
                      "fromLane": str(i),
                      "toLane": str(i)}]
            if not self.one_way:
                conn += [{"from": "top_center",
                        "to": "center_bottom",
                        "fromLane": str(i),
                        "toLane": str(i)}]
                conn += [{"from": "right_center",
                        "to": "center_left",
                        "fromLane": str(i),
                        "toLane": str(i)}]
            if self.turns:
                # This network has specific turns
                conn += [{"from": "left_center",
                          "to": "center_top",
                          "fromLane": str(i),
                          "toLane": str(i)}]
                conn += [{"from": "bottom_center",
                          "to": "center_right",
                          "fromLane": str(i),
                          "toLane": str(i)}]
                if not self.one_way:
                    conn += [{"from": "left_center",
                            "to": "center_bottom",
                            "fromLane": str(i),
                            "toLane": str(i)}]
                    conn += [{"from": "bottom_center",
                            "to": "center_left",
                            "fromLane": str(i),
                            "toLane": str(i)}]
                    conn += [{"from": "right_center",
                            "to": "center_top",
                            "fromLane": str(i),
                            "toLane": str(i)}]
                    conn += [{"from": "top_center",
                            "to": "center_right",
                            "fromLane": str(i),
                            "toLane": str(i)}]
                    conn += [{"from": "right_center",
                            "to": "center_bottom",
                            "fromLane": str(i),
                            "toLane": str(i)}]
                    conn += [{"from": "top_center",
                            "to": "center_left",
                            "fromLane": str(i),
                            "toLane": str(i)}]
        return { "center": conn }

    def specify_edge_starts(self):
        """See base class."""
        edgestarts = [
            ("bottom_center", self.edge_length),
            ("center_top", 2 * self.edge_length),
            ("center_right", 3 * self.edge_length),
            ("left_center", 4 * self.edge_length)]
        if not self.one_way:
            edgestarts += [("center_bottom", 5 * self.edge_length),
                           ("top_center", 6 * self.edge_length),
                           ("right_center", 7 * self.edge_length),
                           ("center_left", 8 * self.edge_length)]

        return edgestarts

    def specify_intersection_edge_starts(self):
        """See parent class."""
        intersection_edgestarts = \
            [(":center", 0)]
        return intersection_edgestarts
