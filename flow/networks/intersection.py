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
    "edge_length": 100
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
            "id": "bottom",
            "type": "edgeType",
            "priority": "78",
            "from": "bottom",
            "to": "center",
            "length": self.edge_length
        }, {
            "id": "top",
            "type": "edgeType",
            "priority": 78,
            "from": "center",
            "to": "top",
            "length": self.edge_length
        }, {
            "id": "right",
            "type": "edgeType",
            "priority": 46,
            "from": "center",
            "to": "right",
            "length": self.edge_length
        }, {
            "id": "left",
            "type": "edgeType",
            "priority": 46,
            "from": "left",
            "to": "center",
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
        """See parent class.
        
        
        """
        if self.turns:
            # Probability that cars coming from the bottom road will turn right onto right road
            prob_right = 0.5
            # Probability that cars coming from the left road will turn left onto top road
            prob_left = 0.5
            rts = {
                "bottom": [(["bottom", "top"], 1-prob_right),
                        (["bottom", "right"], prob_right)],
                "left": [(["left", "right"], 1-prob_left),
                        (["left", "top"], prob_left)]              
            }
        else:
            # Do no turns
            rts = {
                "bottom":["bottom", "top"],
                "left": ["left", "right"]
            }
        return rts

    def specify_connections(self, net_params):
        """See parent class."""
        conn = []
        for i in range(self.lanes):
            conn += [{"from": "bottom",
                      "to": "top",
                      "fromLane": str(i),
                      "toLane": str(i)}]
            conn += [{"from": "left",
                      "to": "right",
                      "fromLane": str(i),
                      "toLane": str(i)}]
            if self.turns:
                # This network has specific turns
                conn += [{"from": "left",
                          "to": "top",
                          "fromLane": str(i),
                          "toLane": str(i)}]
                conn += [{"from": "bottom",
                          "to": "right",
                          "fromLane": str(i),
                          "toLane": str(i)}]       
        return { "center": conn }

    def specify_edge_starts(self):
        """See base class."""
        edgestarts = [
            ("bottom", self.edge_length),
            ("top", 2 * self.edge_length),
            ("right", 3 * self.edge_length),
            ("left", 4 * self.edge_length)]

        return edgestarts

    def specify_intersection_edge_starts(self):
        """See parent class."""
        intersection_edgestarts = \
            [(":center", 0)]
        return intersection_edgestarts