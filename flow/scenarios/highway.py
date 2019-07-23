"""Contains the highway scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # length of the highway
    "length": 1000,
    # number of lanes
    "lanes": 4,
    # speed limit for all edges
    "speed_limit": 30,
    # number of edges to divide the highway into
    "num_edges": 1
}


class HighwayScenario(Scenario):
    """Highway scenario class.

    This network consists of `num_edges` different straight highway sections
    with a total characteristic length and number of lanes.

    Requires from net_params:

    * **length** : length of the highway
    * **lanes** : number of lanes in the highway
    * **speed_limit** : max speed limit of the highway
    * **num_edges** : number of edges to divide the highway into

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.scenarios import HighwayScenario
    >>>
    >>> scenario = HighwayScenario(
    >>>     name='highway',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'length': 230,
    >>>             'lanes': 1,
    >>>             'speed_limit': 30,
    >>>             'num_edges': 1
    >>>         },
    >>>         no_internal_links=True  # we do not want junctions
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a highway scenario."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.length = net_params.additional_params["length"]
        self.lanes = net_params.additional_params["lanes"]
        self.num_edges = net_params.additional_params.get("num_edges", 1)

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        num_edges = net_params.additional_params.get("num_edges", 1)
        segment_lengths = np.linspace(0, length, num_edges+1)

        nodes = []
        for i in range(num_edges+1):
            nodes += [{
                "id": "edge_{}".format(i),
                "x": segment_lengths[i],
                "y": 0
            }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        num_edges = net_params.additional_params.get("num_edges", 1)
        segment_length = length/float(num_edges)

        edges = []
        for i in range(num_edges):
            edges += [{
                "id": "highway_{}".format(i),
                "type": "highwayType",
                "from": "edge_{}".format(i),
                "to": "edge_{}".format(i+1),
                "length": segment_length
            }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]

        types = [{
            "id": "highwayType",
            "numLanes": lanes,
            "speed": speed_limit
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        num_edges = net_params.additional_params.get("num_edges", 1)
        rts = {}
        for i in range(num_edges):
            rts["highway_{}".format(i)] = ["highway_{}".format(j) for
                                           j in range(i, num_edges)]

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        edgestarts = [("highway_{}".format(i), 0)
                      for i in range(self.num_edges)]
        return edgestarts

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """Generate a user defined set of starting positions.

        This method is just used for testing.
        """
        return initial_config.additional_params["start_positions"], \
            initial_config.additional_params["start_lanes"]
