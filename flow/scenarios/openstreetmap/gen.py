"""Contains the generator class for OpenStreetMap files."""

from flow.core.generator import Generator

import sys
import subprocess


class OpenStreetMapGenerator(Generator):
    """Class used to generate network files from an OpenStreetMap (.osm) file.

    The .osm file is specified in the NetParams object. For example:

        >>> from flow.core.params import NetParams
        >>> net_params = NetParams(osm_path="/path/to/osm_file.osm")

    No "specify_nodes" and "specify_edges" routes are needed. However, a
    "specify_routes" file is still needed to specify the appropriate routes
    vehicles can traverse in the network.
    """

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
