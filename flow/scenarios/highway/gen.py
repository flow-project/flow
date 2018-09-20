"""Contains the highway scenario class."""

from flow.core.generator import Generator
import numpy as np


class HighwayGenerator(Generator):
    """Generator for multi-lane highways."""

    def __init__(self, net_params, base):
        """Instantiate a generator class for highways.

        See parent class for description of parameters.
        """
        length = net_params.additional_params["length"]
        lanes = net_params.additional_params["lanes"]
        self.name = "%s-%dm%dl" % (base, length, lanes)

        super().__init__(net_params, base)

    def specify_nodes(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        num_edges = net_params.additional_params.get("num_edges", 1)
        segment_lengths = np.linspace(0, length, num_edges+1)

        nodes = []
        for i in range(num_edges+1):
            nodes += [{
                "id": "edge_{}".format(i),
                "x": repr(segment_lengths[i]),
                "y": repr(0)
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
                "length": repr(segment_length)
            }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]

        types = [{
            "id": "highwayType",
            "numLanes": repr(lanes),
            "speed": repr(speed_limit)
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
