"""Contains the scenario class for OpenStreetMap files."""

from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.scenarios.base_scenario import Scenario
import sys
import subprocess


class OpenStreetMapScenario(Scenario):

    def specify_nodes(self, net_params):
        """See class definition."""
        pass

    def specify_edges(self, net_params):
        """See class definition."""
        pass

    def specify_edge_starts(self):
        """See parent class.

        The edge starts are specified from the network configuration file. Note
        that, the values are arbitrary but do not allow the positions of any
        two edges to overlap, thereby making them compatible with all starting
        position methods for vehicles.
        """
        # the total length of the network is defined within this function
        self.length = 0

        edgestarts = []
        for edge_id in sorted(self._edge_list):
            # the current edge starts where the last edge ended
            edgestarts.append((edge_id, self.length))
            # increment the total length of the network with the length of the
            # current edge
            self.length += self._edges[edge_id]["length"]

        return edgestarts

    def specify_internal_edge_starts(self):
        """See parent class.

        All internal edge starts are given a position of -1. This may be
        overridden; however, in general we do not worry about internal edges
        and junctions in large networks.
        """
        return [(":", -1)]
