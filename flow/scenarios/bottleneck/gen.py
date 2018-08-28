"""Contains the bottleneck generator class."""

from flow.core.generator import Generator
import numpy as np


class BottleneckGenerator(Generator):
    """Generator class for simulating a bottleneck.

    No parameters are needed from net_params (the network is not parametrized).
    """

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
