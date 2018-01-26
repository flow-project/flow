from flow.core.generator import Generator

import sys
import subprocess
import os
from lxml import etree
import xml.etree.ElementTree as ET

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

        # collect edge data from the generated network configuration file
        edges_dict = self._import_edges_from_net()

        return edges_dict

    def specify_nodes(self, net_params):
        pass

    def specify_edges(self, net_params):
        pass

    def _import_edges_from_net(self):
        """
        Imports a network configuration file, and returns the information on the
        edges located in the file.

        Return
        ------
        net_data: dict <dict>
            Key = name of the edge
            Element = num_lanes, speed_limit, length
        """

        # import the .net.xml file containing all edge/type data
        parser = etree.XMLParser(recover=True)
        tree = ET.parse(os.path.join(self.net_params.cfg_path, self.netfn),
                        parser=parser)
        root = tree.getroot()

        # Collect information on the available types (if any are available).
        # This may be used when specifying some edge data.
        types_data = dict()

        for typ in root.findall('type'):
            type_id = typ.attrib["id"]
            types_data[type_id] = dict()

            if "speed" in typ.attrib:
                types_data[type_id]["speed"] = typ.attrib["speed"]
            else:
                types_data[type_id]["speed"] = None

            if "numLanes" in typ.attrib:
                types_data[type_id]["numLanes"] = typ.attrib["numLanes"]
            else:
                types_data[type_id]["numLanes"] = None

        # collect all information on the edges
        net_data = dict()

        for edge in root.findall('edge'):
            edge_id = edge.attrib["id"]

            # skip an edge if it is an internal link / junction
            if edge_id[0] == ":":
                continue

            # create a new key for this edge
            net_data[edge_id] = dict()

            # check for speed
            if "speed" in edge:
                net_data[edge_id]["speed"] = float(edge.attrib["speed"])
            else:
                net_data[edge_id]["speed"] = None

            # if the edge has a type parameters, check that type for a speed and
            # parameter if one was not already found
            if "type" in edge.attrib and edge.attrib["type"] in types_data:
                if net_data[edge_id]["speed"] is None:
                    net_data[edge_id]["speed"] = \
                        float(types_data[edge.attrib["type"]]["speed"])

            # collect the length from the lane sub-element in the edge, the
            # number of lanes from the number of lane elements, and if needed,
            # also collect the speed value (assuming it is there)
            net_data[edge_id]["lanes"] = 0
            for i, lane in enumerate(edge):
                net_data[edge_id]["lanes"] += 1
                if i == 0:
                    net_data[edge_id]["length"] = float(lane.attrib["length"])
                    if net_data[edge_id]["speed"] is None \
                            and "speed" in lane.attrib:
                        net_data[edge_id]["speed"] = lane.attrib["speed"]

            # if no speed value is present anywhere, set it to some default
            if net_data[edge_id]["speed"] is None:
                net_data[edge_id]["speed"] = 30

        return net_data
