"""Contains the loop merge scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos, linspace

ADDITIONAL_NET_PARAMS = {
    # radius of the loops
    "ring_radius": 50,
    # length of the straight edges connected the outer loop to the inner loop
    "lane_length": 75,
    # number of lanes in the inner loop
    "inner_lanes": 3,
    # number of lanes in the outer loop
    "outer_lanes": 2,
    # max speed limit in the network
    "speed_limit": 30,
    # resolution of the curved portions
    "resolution": 40,
}


class TwoLoopsOneMergingScenario(Scenario):
    """Two loop merge scenario.

    This network is expected to simulate a closed loop representation of a
    merge. It consists of two rings that merge together for half the length of
    the smaller ring.

    Requires from net_params:

    * **ring_radius** : radius of the loops
    * **lane_length** : length of the straight edges connected the outer loop
      to the inner loop
    * **inner_lanes** : number of lanes in the inner loop
    * **outer_lanes** : number of lanes in the outer loop
    * **speed_limit** : max speed limit in the network
    * **resolution** : resolution of the curved portions

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.scenarios import TwoLoopsOneMergingScenario
    >>>
    >>> scenario = TwoLoopsOneMergingScenario(
    >>>     name='two_loops_merge',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'ring_radius': 50,
    >>>             'lane_length': 75,
    >>>             'inner_lanes': 3,
    >>>             'outer_lanes': 2,
    >>>             'speed_limit': 30,
    >>>             'resolution': 40
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
        """Initialize a two loop scenario."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        radius = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]

        self.inner_lanes = net_params.additional_params["inner_lanes"]
        self.outer_lanes = net_params.additional_params["outer_lanes"]

        self.junction_length = 0.3
        self.intersection_length = 25.5  # calibrate when the radius changes

        net_params.additional_params["length"] = \
            2 * x + 2 * pi * radius + \
            2 * self.intersection_length + 2 * self.junction_length

        num_vehicles = vehicles.num_vehicles
        num_merge_vehicles = sum("merge" in vehicles.get_type(veh_id)
                                 for veh_id in vehicles.ids)
        self.n_inner_vehicles = num_merge_vehicles
        self.n_outer_vehicles = num_vehicles - num_merge_vehicles

        radius = net_params.additional_params["ring_radius"]
        length_loop = 2 * pi * radius
        self.length_loop = length_loop

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        r = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]

        nodes = [{
            "id": "top_left",
            "x": 0,
            "y": r,
            "type": "priority"
        }, {
            "id": "bottom_left",
            "x": 0,
            "y": -r,
            "type": "priority"
        }, {
            "id": "top_right",
            "x": x,
            "y": r,
            "type": "priority"
        }, {
            "id": "bottom_right",
            "x": x,
            "y": -r,
            "type": "priority"
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        r = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]

        ring_edgelen = pi * r
        resolution = 40

        edges = [{
            "id":
            "center",
            "from":
            "bottom_left",
            "to":
            "top_left",
            "type":
            "edgeType",
            "length":
            ring_edgelen,
            "priority":
            46,
            "shape":
            [
                (r * cos(t), r * sin(t))
                for t in linspace(-pi / 2, pi / 2, resolution)
            ],
            "numLanes":
            self.inner_lanes
        }, {
            "id": "top",
            "from": "top_right",
            "to": "top_left",
            "type": "edgeType",
            "length": x,
            "priority": 46,
            "numLanes": self.outer_lanes
        }, {
            "id": "bottom",
            "from": "bottom_left",
            "to": "bottom_right",
            "type": "edgeType",
            "length": x,
            "numLanes": self.outer_lanes
        }, {
            "id":
            "left",
            "from":
            "top_left",
            "to":
            "bottom_left",
            "type":
            "edgeType",
            "length":
            ring_edgelen,
            "shape":
            [
                (r * cos(t), r * sin(t))
                for t in linspace(pi / 2, 3 * pi / 2, resolution)
            ],
            "numLanes":
            self.inner_lanes
        }, {
            "id":
            "right",
            "from":
            "bottom_right",
            "to":
            "top_right",
            "type":
            "edgeType",
            "length":
            ring_edgelen,
            "shape":
            [
                (x + r * cos(t), r * sin(t))
                for t in linspace(-pi / 2, pi / 2, resolution)
            ],
            "numLanes":
            self.outer_lanes
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        speed_limit = net_params.additional_params["speed_limit"]

        types = [{"id": "edgeType", "speed": speed_limit}]
        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "top": ["top", "left", "bottom", "right", "top"],
            "bottom": ["bottom", "right", "top", "left", "bottom"],
            "right": ["right", "top", "left", "bottom"],
            "left": ["left", "center", "left"],
            "center": ["center", "left", "center"]
        }

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        r = self.net_params.additional_params["ring_radius"]
        lane_length = self.net_params.additional_params["lane_length"]

        ring_edgelen = pi * r

        edgestarts = [
            ("left", self.intersection_length),
            ("center", ring_edgelen + 2 * self.intersection_length),
            ("bottom", 2 * ring_edgelen + 2 * self.intersection_length),
            ("right", 2 * ring_edgelen + lane_length +
             2 * self.intersection_length + self.junction_length),
            ("top", 3 * ring_edgelen + lane_length +
             2 * self.intersection_length + 2 * self.junction_length)
        ]

        return edgestarts

    def specify_internal_edge_starts(self):
        """See parent class."""
        r = self.net_params.additional_params["ring_radius"]
        lane_length = self.net_params.additional_params["lane_length"]

        ring_edgelen = pi * r

        internal_edgestarts = [
            (":top_left", 0), (":bottom_left",
                               ring_edgelen + self.intersection_length),
            (":bottom_right",
             2 * ring_edgelen + lane_length + 2 * self.intersection_length),
            (":top_right", 3 * ring_edgelen + lane_length +
             2 * self.intersection_length + self.junction_length)
        ]

        return internal_edgestarts
