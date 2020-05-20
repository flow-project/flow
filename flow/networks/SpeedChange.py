"""Contains the highway network class."""

from flow.networks.base import Network
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
    # end speed limit
    "end_speed_limit": 25,
    # number of edges to divide the highway into
    "num_edges": 1,
    # Length of the cell imposing a boundary
    "boundary_cell_length": 500
}


class HighwayNetwork_Modified(Network):
    """Highway network class.

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
    >>> from flow.networks import HighwayNetwork
    >>>
    >>> network = HighwayNetwork(
    >>>     name='highway',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'length': 230,
    >>>             'lanes': 1,
    >>>             'speed_limit': 30,
    >>>             'num_edges': 1
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
        """Initialize a highway network."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.length = net_params.additional_params["length"]
        self.lanes = net_params.additional_params["lanes"]
        self.num_edges = net_params.additional_params.get("num_edges", 1)
        self.end_length = net_params.additional_params["boundary_cell_length"]

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

        nodes +=[{"id": "edge_{}".format(num_edges+1),
                "x": length+self.end_length,
                "y": 0}]

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

        edges += [{
                "id": "highway_end",
                "type": "highway_end",
                "from": "edge_{}".format(num_edges),
                "to": "edge_{}".format(num_edges+1),
                "length": self.end_length
            }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        end_speed_limt = net_params.additional_params["end_speed_limit"]

        types = [{
            "id": "highwayType",
            "numLanes": lanes,
            "speed": speed_limit
        }]

        types +=[{
            "id":"highway_end",
            "numLanes":lanes,
            "speed":end_speed_limt}]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        num_edges = net_params.additional_params.get("num_edges", 1)
        rts = {}
        for i in range(num_edges):
            rts["highway_{}".format(i)] = ["highway_{}".format(j) for
                                           j in range(i, num_edges)]
            rts["highway_{}".format(i)].append("highway_end")

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        num_edges = self.num_edges
        edgestarts = [("highway_{}".format(i), 0)
                      for i in range(num_edges)]


        # Adding this line fixes the problem that the simulation breaks when it reaches
        # the slow down segment, but then the simulation doesn't advance past the first step.

        edgestarts += [("highway_end",self.length)]
        return edgestarts

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """Generate a user defined set of starting positions.

        This method is just used for testing.
        """
        return initial_config.additional_params["start_positions"], \
            initial_config.additional_params["start_lanes"]
