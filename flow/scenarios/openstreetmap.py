"""Contains the scenario class for OpenStreetMap files."""

from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.scenarios.base_scenario import Scenario
import sys
import subprocess


class OpenStreetMapScenario(Scenario):
    """Class used to generate network files from an OpenStreetMap (.osm) file.

    The .osm file is specified in the NetParams object. For example:

        >>> from flow.core.params import NetParams
        >>> net_params = NetParams(osm_path="/path/to/osm_file.osm")

    No "specify_nodes" and "specify_edges" routes are needed. However, a
    "specify_routes" file is still needed to specify the appropriate routes
    vehicles can traverse in the network.
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initialize a scenario from a .osm file.

        See flow/scenarios/base_scenario.py for description of params.
        """
        if net_params.osm_path is None:
            raise ValueError("Path to the OpenStreetMap file must be specified"
                             " in net_params.")

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def generate_net(self, net_params, traffic_lights):
        """See parent class.

        The network file is generated from the .osm file specified in
        net_params.osm_path
        """
        # specify the location of the input osm file
        osm_path = net_params.osm_path

        # specify the location of the output file
        netfn = "%s.net.xml" % self.name

        # generate the network file with sumo
        net_cmd = "netconvert --osm-files {0} --output-file {1}".\
            format(osm_path, self.cfg_path + netfn)

        # this handles removing all roads in the network that cannot be ridden
        # by vehicles
        net_cmd += \
            " --remove-edges.by-vclass rail_slow,rail_fast,bicycle,pedestrian"

        # this removes edges that are not connected to a network (isolated)
        net_cmd += " --remove-edges.isolated"

        # this removes internal links from the network (useful when the network
        # becomes very large)
        if net_params.no_internal_links:
            net_cmd += " --no_internal_links"

        subprocess.call(
            net_cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)

        # name of the .net.xml file (located in cfg_path)
        self.netfn = netfn

        # collect data from the generated network configuration file
        edges_dict, conn_dict = self._import_edges_from_net()

        return edges_dict, conn_dict

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
        for edge_id in self._edge_list:
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
