from cistar.core.generator import Generator
from cistar.controllers.base_controller import SumoController

from cistar.core.util import makexml
from cistar.core.util import printxml

import subprocess
import sys

from numpy import pi, sin, cos, linspace

import logging
import random
from lxml import etree
E = etree.Element


class BraessParadoxGenerator(Generator):
    """
    Generates Net files for two-way intersection sim. Requires:
    - edge_length: length of any of the four lanes associated with the diamond portion of the network.
    - angle: angle between the horizontal axis and the edges associated with the diamond.
    - resolution:
    - AC_DB_speed_limit: speed limit of the vehicles of the AC and DB links
    - AD_CB_speed_limit: speed limit of the vehicles of the AD and CB links.
    """

    def __init__(self, net_params, net_path, cfg_path, base):
        """
        See parent class
        """
        super().__init__(net_params, net_path, cfg_path, base)

        edge_len = net_params["edge_length"]
        self.name = "%s-%dm" % (base, edge_len)

    def specify_nodes(self, net_params):
        """
        See parent class
        """
        edge_len = net_params["edge_length"]
        angle = net_params["angle"]
        edge_x = edge_len * cos(angle)
        edge_y = edge_len * sin(angle)
        r = 0.75 * edge_y

        nodes = [{"id": "A",   "x": repr(0),          "y": repr(0),       "type": "unregulated"},
                 {"id": "C",   "x": repr(edge_x),     "y": repr(edge_y),  "type": "unregulated"},
                 {"id": "D",   "x": repr(edge_x),     "y": repr(-edge_y), "type": "unregulated"},
                 {"id": "B",   "x": repr(2 * edge_x), "y": repr(0),       "type": "unregulated"},
                 {"id": "BA1", "x": repr(2 * edge_x), "y": repr(-2 * r),  "type": "unregulated"},
                 {"id": "BA2", "x": repr(0),          "y": repr(-2 * r),  "type": "unregulated"}]

        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        edge_len = net_params["edge_length"]
        AC_DB_speed_limit = net_params["AC_DB_speed_limit"]
        AD_CB_speed_limit = net_params["AD_CB_speed_limit"]
        resolution = net_params["resolution"]
        angle = net_params["angle"]

        edge_x = edge_len * cos(angle)
        edge_y = edge_len * sin(angle)
        r = 0.75 * edge_y
        curve_len = r * pi
        straight_horz_len = 2 * edge_x

        # braess network component, consisting of the following edges:
        # - AC: top-left portion of the diamond
        # - AD: bottom-left portion of the diamond
        # - CB: top-right portion of the diamond
        # - DB: bottom-right portion of the diamond
        # - CD: vertical edge connecting lanes AC and DB
        edges = [{"id": "AC", "from": "A", "to": "C", "numLanes": "2", "length": repr(edge_len),
                  "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit))},
                 {"id": "AD", "from": "A", "to": "D", "numLanes": "1", "length": repr(edge_len),
                  "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit))},
                 {"id": "CB", "from": "C", "to": "B", "numLanes": "1", "length": repr(edge_len),
                  "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit))},
                 {"id": "CD", "from": "C", "to": "D", "numLanes": "1", "length": repr(2 * edge_y),
                  "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit))},
                 {"id": "DB", "from": "D", "to": "B", "numLanes": "2", "length": repr(edge_len),
                  "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit))}]

        # connecting output to input in braess network (to produce loop)
        # Edges B and BA2 produce the two semi-circles on either sides of the braess network,
        # while edge BA1 is a straight line that connects these to semicircles.
        edges += [{"id": "B", "from": "B", "to": "BA1", "numLanes": "3", "length": repr(curve_len),
                   "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit)),
                   "shape": " ".join(["%.2f,%.2f" % (2 * edge_x + r * sin(t), r * (- 1 + cos(t)))
                                      for t in linspace(0, pi, resolution)])},
                  {"id": "BA1", "from": "BA1", "to": "BA2", "numLanes": "3",
                   "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit)), "length": repr(straight_horz_len)},
                  {"id": "BA2", "from": "BA2", "to": "A", "numLanes": "3", "length": repr(curve_len),
                   "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit)),
                   "shape": " ".join(["%.2f,%.2f" % (- r * sin(t), - r * (1 + cos(t)))
                                      for t in linspace(0, pi, resolution)])}]

        return edges

    def specify_types(self, net_params):
        """
        See parent class
        """
        lanes = net_params["lanes"]
        AC_DB_speed_limit = net_params["AC_DB_speed_limit"]
        AD_CB_speed_limit = net_params["AD_CB_speed_limit"]

        types = [{"id": "edgeType", "numLanes": repr(lanes), "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit))}]

        return types

    def specify_connections(self, net_params):
        """
        See parent class
        """
        connections = [{"from": "AC", "to": "CB", "fromLane": "1", "toLane": "0"},
                       {"from": "AC", "to": "CD", "fromLane": "0", "toLane": "0"}]

        return connections

    def specify_routes(self, net_params):
        """
        See parent class
        """
        rts = {"AC":  ["AC", "CB", "B", "BA1", "BA2"],
               "AD":  ["AD", "DB", "B", "BA1", "BA2"],
               "CB":  ["CB", "B", "BA1", "BA2"],
               "CD":  ["CD", "DB", "B", "BA1", "BA2"],
               "DB":  ["DB", "B", "BA1", "BA2"],
               "B":   ["B", "BA1", "BA2"],
               "BA1": ["BA1", "BA2"],
               "BA2": ["BA2", "AC", "CB"]}
        return rts
