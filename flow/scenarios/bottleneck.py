"""Contains the bottleneck scenario class."""

from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.scenarios.base_scenario import Scenario
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # the factor multiplying number of lanes.
    "scaling": 1,
}


class BottleneckScenario(Scenario):
    """Scenario class for bottleneck simulations."""

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Instantiate the scenario class.

        Requires from net_params:
        - scaling: the factor multiplying number of lanes

        In order for right-of-way dynamics to take place at the intersection,
        set "no_internal_links" in net_params to False.

        See flow/scenarios/base_scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        nodes = [
            {
                "id": "1",
                "x": "0",
                "y": "0"
            },  # pre-toll
            {
                "id": "2",
                "x": "100",
                "y": "0"
            },  # toll
            {
                "id": "3",
                "x": "410",
                "y": "0"
            },  # light
            {
                "id": "4",
                "x": "550",
                "y": "0",
                "type": "zipper",
                "radius": "20"
            },  # merge1
            {
                "id": "5",
                "x": "830",
                "y": "0",
                "type": "zipper",
                "radius": "20"
            },  # merge2
            {
                "id": "6",
                "x": "985",
                "y": "0"
            }
        ]  # post-merge2
        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        scaling = net_params.additional_params.get("scaling", 1)
        assert (isinstance(scaling, int)), "Scaling must be an int"

        edges = [
            {
                "id": "1",
                "from": "1",
                "to": "2",
                "length": "100",  #
                "spreadType": "center",
                "numLanes": str(4 * scaling),
                "speed": "23"
            },
            {
                "id": "2",
                "from": "2",
                "to": "3",
                "length": "310",  # DONE
                "spreadType": "center",
                "numLanes": str(4 * scaling),
                "speed": "23"
            },
            {
                "id": "3",
                "from": "3",
                "to": "4",
                "length": "140",  # DONE
                "spreadType": "center",
                "numLanes": str(4 * scaling),
                "speed": "23"
            },
            {
                "id": "4",
                "from": "4",
                "to": "5",
                "length": "280",  # DONE
                "spreadType": "center",
                "numLanes": str(2 * scaling),
                "speed": "23"
            },
            {
                "id": "5",
                "from": "5",
                "to": "6",
                "length": "155",
                "spreadType": "center",
                "numLanes": str(scaling),
                "speed": "23"
            }
        ]

        return edges

    def specify_connections(self, net_params):
        """See parent class."""
        scaling = net_params.additional_params.get("scaling", 1)
        conn = []
        for i in range(4 * scaling):
            conn += [{
                "from": "3",
                "to": "4",
                "fromLane": str(i),
                "toLane": str(int(np.floor(i / 2)))
            }]
        for i in range(2 * scaling):
            conn += [{
                "from": "4",
                "to": "5",
                "fromLane": str(i),
                "toLane": str(int(np.floor(i / 2)))
            }]
        return conn

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "1": ["1", "2", "3", "4", "5"],
            "2": ["2", "3", "4", "5"],
            "3": ["3", "4", "5"],
            "4": ["4", "5"],
            "5": ["5"]
        }

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        return [("1", 0), ("2", 100), ("3", 405), ("4", 425), ("5", 580)]

    def get_bottleneck_lanes(self, lane):
        """Return the reduced number of lanes."""
        return [int(lane / 2), int(lane / 4)]
