from flow.core.params import InitialConfig
from flow.scenarios.base_scenario import Scenario


class OpenStreetMapScenario(Scenario):

    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=InitialConfig()):
        """
        Initializes a scenario from a .osm file.

        Required net_params: osm_path
        See Scenario.py for description of params.
        """
        if net_params.osm_path is None:
            raise ValueError("Path to the OpenStreetMap file must be specified "
                             "in net_params.")

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config=initial_config)

    def specify_edge_starts(self):
        """
        See parent class.

        The edge starts are specified from the network configuration file. Note
        that, the values are arbitrary but do not allow the positions of any two
        edges to overlap, thereby making them compatible with all starting
        position methods for vehicles.
        """
        # the total length of the network is defined within this function
        self.length = 0

        edgestarts = []
        for edge_id in self.edges:
            # the current edge starts (in 1D position) where the last edge ended
            edgestarts.append((edge_id, self.length))
            # increment the total length of the network with the length of the
            # current edge
            self.length += self.edges[edge_id]["length"]

        return edgestarts

    def specify_internal_edge_starts(self):
        """
        See parent class.

        All internal edge starts are given a position of -1. This may be
        overridden; however, in general we do not worry about internal edges and
        junctions in large networks.
        """
        return [(":", -1)]
