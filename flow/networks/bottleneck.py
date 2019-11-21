"""Contains the bottleneck network class."""

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks.base import Network
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # the factor multiplying number of lanes.
    "scaling": 1,
    # edge speed limit
    'speed_limit': 23
}


class BottleneckNetwork(Network):
    """Network class for bottleneck simulations.

    This network acts as a scalable representation of the Bay Bridge. It
    consists of a two-stage lane-drop bottleneck where 4n lanes reduce to 2n
    and then to n, where n is the scaling value. The length of the bottleneck
    is fixed.

    Requires from net_params:

    * **scaling** : the factor multiplying number of lanes
    * **speed_limit** : edge speed limit

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import BottleneckNetwork
    >>>
    >>> network = BottleneckNetwork(
    >>>     name='bottleneck',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'scaling': 1,
    >>>             'speed_limit': 1,
    >>>         },
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Instantiate the network class."""
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
                "x": 0,
                "y": 0
            },  # pre-toll
            {
                "id": "2",
                "x": 100,
                "y": 0
            },  # toll
            {
                "id": "3",
                "x": 410,
                "y": 0
            },  # light
            {
                "id": "4",
                "x": 550,
                "y": 0,
                "type": "zipper",
                "radius": 20
            },  # merge1
            {
                "id": "5",
                "x": 830,
                "y": 0,
                "type": "zipper",
                "radius": 20
            },  # merge2
            {
                "id": "6",
                "x": 985,
                "y": 0
            },
            # fake nodes used for visualization
            {
                "id": "fake1",
                "x": 0,
                "y": 1
            },
            {
                "id": "fake2",
                "x": 0,
                "y": 2
            }
        ]  # post-merge2
        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        scaling = net_params.additional_params.get("scaling", 1)
        speed = net_params.additional_params['speed_limit']
        assert (isinstance(scaling, int)), "Scaling must be an int"

        edges = [
            {
                "id": "1",
                "from": "1",
                "to": "2",
                "length": 100,
                "spreadType": "center",
                "numLanes": 4 * scaling,
                "speed": speed
            },
            {
                "id": "2",
                "from": "2",
                "to": "3",
                "length": 310,
                "spreadType": "center",
                "numLanes": 4 * scaling,
                "speed": speed
            },
            {
                "id": "3",
                "from": "3",
                "to": "4",
                "length": 140,
                "spreadType": "center",
                "numLanes": 4 * scaling,
                "speed": speed
            },
            {
                "id": "4",
                "from": "4",
                "to": "5",
                "length": 280,
                "spreadType": "center",
                "numLanes": 2 * scaling,
                "speed": speed
            },
            {
                "id": "5",
                "from": "5",
                "to": "6",
                "length": 155,
                "spreadType": "center",
                "numLanes": scaling,
                "speed": speed
            },
            # fake edge used for visualization
            {
                "id": "fake_edge",
                "from": "fake1",
                "to": "fake2",
                "length": 1,
                "spreadType": "center",
                "numLanes": scaling,
                "speed": speed
            }
        ]

        return edges

    def specify_connections(self, net_params):
        """See parent class."""
        scaling = net_params.additional_params.get("scaling", 1)
        conn_dic = {}
        conn = []
        for i in range(4 * scaling):
            conn += [{
                "from": "3",
                "to": "4",
                "fromLane": i,
                "toLane": int(np.floor(i / 2))
            }]
        conn_dic["4"] = conn
        conn = []
        for i in range(2 * scaling):
            conn += [{
                "from": "4",
                "to": "5",
                "fromLane": i,
                "toLane": int(np.floor(i / 2))
            }]
        conn_dic["5"] = conn
        return conn_dic

    def specify_centroids(self, net_params):
        """See parent class."""
        centroids = []
        centroids += [{
            "id": "1",
            "from": None,
            "to": "1",
            "x": -30,
            "y": 0,
        }]
        centroids += [{
            "id": "1",
            "from": "5",
            "to": None,
            "x": 985 + 30,
            "y": 0,
        }]
        return centroids

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
