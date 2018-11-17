"""Contains the ring road scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from numpy import pi, sin, cos, linspace

ADDITIONAL_NET_PARAMS = {
    # length of the ring road
    "length": 230,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # resolution of the curves on the ring
    "resolution": 40
}


class LoopScenario(Scenario):
    """Ring road scenario."""

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initialize a loop scenario.

        Requires from net_params:
        - length: length of the circle
        - lanes: number of lanes in the circle
        - speed_limit: max speed limit of the circle
        - resolution: number of nodes resolution

        See flow/scenarios/base_scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.length = net_params.additional_params["length"]
        self.lanes = net_params.additional_params["lanes"]

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        r = length / (2 * pi)

        nodes = [{
            "id": "bottom",
            "x": repr(0),
            "y": repr(-r)
        }, {
            "id": "right",
            "x": repr(r),
            "y": repr(0)
        }, {
            "id": "top",
            "x": repr(0),
            "y": repr(r)
        }, {
            "id": "left",
            "x": repr(-r),
            "y": repr(0)
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        resolution = net_params.additional_params["resolution"]
        r = length / (2 * pi)
        edgelen = length / 4.

        edges = [{
            "id":
            "bottom",
            "type":
            "edgeType",
            "from":
            "bottom",
            "to":
            "right",
            "length":
            repr(edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * cos(t), r * sin(t))
                for t in linspace(-pi / 2, 0, resolution)
            ])
        }, {
            "id":
            "right",
            "type":
            "edgeType",
            "from":
            "right",
            "to":
            "top",
            "length":
            repr(edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * cos(t), r * sin(t))
                for t in linspace(0, pi / 2, resolution)
            ])
        }, {
            "id":
            "top",
            "type":
            "edgeType",
            "from":
            "top",
            "to":
            "left",
            "length":
            repr(edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * cos(t), r * sin(t))
                for t in linspace(pi / 2, pi, resolution)
            ])
        }, {
            "id":
            "left",
            "type":
            "edgeType",
            "from":
            "left",
            "to":
            "bottom",
            "length":
            repr(edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * cos(t), r * sin(t))
                for t in linspace(pi, 3 * pi / 2, resolution)
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
            "top": ["top", "left", "bottom", "right"],
            "left": ["left", "bottom", "right", "top"],
            "bottom": ["bottom", "right", "top", "left"],
            "right": ["right", "top", "left", "bottom"]
        }

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        edgelen = self.length / 4

        edgestarts = [("bottom", 0), ("right", edgelen), ("top", 2 * edgelen),
                      ("left", 3 * edgelen)]

        return edgestarts
