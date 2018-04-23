from flow.core.generator import Generator
from lxml import etree

E = etree.Element


class NetFileGenerator(Generator):
    """
    A class used to generate network configuration files from an OpenStreetMap
    (.osm) file.

    The .osm file is specified in the net_params.osm_path file.

    No "specify_nodes" and "specify_edges" routes are needed. However, a
    "specify_routes" file is still needed to specify the appropriate routes
    vehicles can traverse in the network.
    """

    def generate_net(self, net_params, traffic_lights):
        """
        See parent class.
        The network file is generated from the .osm file specified in
        net_params.osm_path
        """
        # name of the .net.xml file (located in cfg_path)
        self.netfn = net_params.netfile

        # collect data from the generated network configuration file
        edges_dict, conn_dict = self._import_edges_from_net()

        return edges_dict, conn_dict

    def specify_nodes(self, net_params):
        pass

    def specify_edges(self, net_params):
        pass
