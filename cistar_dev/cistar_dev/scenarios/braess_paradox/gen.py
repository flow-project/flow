from cistar_dev.core.exp import Generator
from cistar_dev.controllers.base_controller import SumoController

from cistar_dev.core.util import makexml
from cistar_dev.core.util import printxml

import subprocess
import sys

import numpy as np
from numpy import pi, sin, cos, linspace

import logging
import random
from lxml import etree
E = etree.Element


class BraessParadoxGenerator(Generator):

    def generate_net(self, params):
        """
        Generates Net files for two-way intersection sim. Requires:
        - edge_length: length of any of the four lanes associated with the diamond portion of the network.
        - angle: angle between the horizontal axis and the edges associated with the diamond.
        - resolution:
        - AC_DB_speed_limit: max speed limit of the vehicles of the AC and DB links
        - AD_CB_speed_limit: max speed limit of the vehicles of the AD and CB links.
        """
        edge_len = params["edge_length"]
        lanes = params["lanes"]
        AC_DB_speed_limit = params["AC_DB_speed_limit"]
        AD_CB_speed_limit = params["AD_CB_speed_limit"]
        resolution = params["resolution"]
        angle = params["angle"]

        edge_x = edge_len * cos(angle)
        edge_y = edge_len * sin(angle)
        r = 0.75 * edge_y
        curve_len = r * pi
        straight_horz_len = 2 * edge_x
        length = 4 * edge_len + 2 * curve_len + straight_horz_len

        self.name = "%s-%dm%dl" % (self.base, length, lanes)

        nodfn = "%s.nod.xml" % self.name
        edgfn = "%s.edg.xml" % self.name
        typfn = "%s.typ.xml" % self.name
        cfgfn = "%s.netccfg" % self.name
        netfn = "%s.net.xml" % self.name
        confn = "%s.con.xml" % self.name

        # xml file for nodes; contains nodes for the boundary points with respect to the x and y axes
        x = makexml("nodes", "http://sumo.dlr.de/xsd/nodes_file.xsd")
        x.append(E("node", id="A",   x=repr(0),          y=repr(0),       type="unregulated"))
        x.append(E("node", id="C",   x=repr(edge_x),     y=repr(edge_y),  type="unregulated"))
        x.append(E("node", id="D",   x=repr(edge_x),     y=repr(-edge_y), type="unregulated"))
        x.append(E("node", id="B",   x=repr(2 * edge_x), y=repr(0),       type="unregulated"))
        x.append(E("node", id="BA1", x=repr(2 * edge_x), y=repr(-2 * r),  type="unregulated"))
        x.append(E("node", id="BA2", x=repr(0),          y=repr(-2 * r),  type="unregulated"))
        printxml(x, self.net_path + nodfn)

        # xml file for edges

        # braess network component, consisting of the following edges:
        # - AC: top-left portion of the diamond
        # - AD: bottom-left portion of the diamond
        # - CB: top-right portion of the diamond
        # - DB: bottom-right portion of the diamond
        # - CD: vertical edge connecting lanes AC and DB
        x = makexml("edges", "http://sumo.dlr.de/xsd/edges_file.xsd")

        x.append(E("edge", attrib={"id": "AC", "from": "A", "to": "C",
                                   "numLanes": "2", "length": repr(edge_len),
                                   "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit))}))

        x.append(E("edge", attrib={"id": "AD", "from": "A", "to": "D",
                                   "numLanes": "1", "length": repr(edge_len),
                                   "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit))}))

        x.append(E("edge", attrib={"id": "CB", "from": "C", "to": "B",
                                   "numLanes": "1", "length": repr(edge_len),
                                   "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit))}))

        x.append(E("edge", attrib={"id": "CD", "from": "C", "to": "D",
                                   "numLanes": "1", "length": repr(2 * edge_y),
                                   "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit))}))

        x.append(E("edge", attrib={"id": "DB", "from": "D", "to": "B",
                                   "numLanes": "2", "length": repr(edge_len),
                                   "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit))}))

        # connecting output to input in braess network (to produce loop)
        # Edges B and BA2 produce the two semi-circles on either sides of the braess network,
        # while edge BA1 is a straight line that connects these to semicircles.
        x.append(E("edge", attrib={"id": "B", "from": "B", "to": "BA1", "numLanes": "3", "length": repr(curve_len),
                                   "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit)),
                                   "shape": " ".join(["%.2f,%.2f" % (2 * edge_x + r * sin(t), r * (- 1 + cos(t)))
                                                      for t in linspace(0, pi, resolution)])}))

        x.append(E("edge", attrib={"id": "BA1", "from": "BA1", "to": "BA2", "numLanes": "3",
                                   "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit)),
                                   "length": repr(straight_horz_len)}))

        x.append(E("edge", attrib={"id": "BA2", "from": "BA2", "to": "A", "numLanes": "3", "length": repr(curve_len),
                                   "speed": repr(max(AD_CB_speed_limit, AC_DB_speed_limit)),
                                   "shape": " ".join(["%.2f,%.2f" % (- r * sin(t), - r * (1 + cos(t)))
                                                      for t in linspace(0, pi, resolution)])}))

        printxml(x, self.net_path + edgfn)

        # xml for connections: specifies which lanes connect to which in the edges
        x = makexml("connections", "http://sumo.dlr.de/xsd/connections_file.xsd")
        x.append(E("connection", attrib={"from": "AC", "to": "CB", "fromLane": "1", "toLane": "0"}))
        x.append(E("connection", attrib={"from": "AC", "to": "CD", "fromLane": "0", "toLane": "0"}))
        printxml(x, self.net_path + confn)

        # xml file for types; contains the the number of lanes and the speed limit for the lanes
        x = makexml("types", "http://sumo.dlr.de/xsd/types_file.xsd")
        x.append(E("type", id="edgeType", numLanes=repr(lanes), speed=repr(max(AD_CB_speed_limit, AC_DB_speed_limit))))
        printxml(x, self.net_path + typfn)

        # xml file for configuration
        # - specifies the location of all files of interest for sumo
        # - specifies output net file
        # - specifies processing parameters for no internal links and no turnarounds
        x = makexml("configuration", "http://sumo.dlr.de/xsd/netconvertConfiguration.xsd")
        t = E("input")
        t.append(E("node-files", value=nodfn))
        t.append(E("edge-files", value=edgfn))
        t.append(E("type-files", value=typfn))
        t.append(E("connection-files", value=confn))
        x.append(t)
        t = E("output")
        t.append(E("output-file", value=netfn))
        x.append(t)
        t = E("processing")
        t.append(E("no-internal-links", value="true"))
        t.append(E("no-turnarounds", value="true"))
        x.append(t)
        printxml(x, self.net_path + cfgfn)

        # In order to minimize the effect of merges on the performance of the network (specifically the
        # emergence of queues) "--no-internals-links" is given the value ("true"), meaning that vehicles
        # cross a merge immediately.
        retcode = subprocess.call(["netconvert -c " + self.net_path + cfgfn + " --output-file=" +
                                   self.cfg_path + netfn + ' --no-internal-links="false"'],
                                  stdout=sys.stdout, stderr=sys.stderr, shell=True)
        self.netfn = netfn

        return self.net_path + netfn

    """ Lets add everything after here to the base generator class """

    def generate_cfg(self, params):
        """
        Generates .sumo.cfg files using net files and netconvert.
        Requires:
        num_cars: Number of cars to seed the simulation with
           max_speed: max speed of cars
           OR
        type_list: List of types of cars to seed the simulation with

        startTime: time to start the simulation
        endTime: time to end the simulation

        """
        if "start_time" not in params:
            raise ValueError("start_time of circle not supplied")
        else:
            start_time = params["start_time"]

        if "end_time" in params:
            end_time = params["end_time"]
        else:
            end_time = None

        self.roufn = "%s.rou.xml" % self.name
        addfn = "%s.add.xml" % self.name
        cfgfn = "%s.sumo.cfg" % self.name
        guifn = "%s.gui.cfg" % self.name

        self.rts = self.available_route_choices()

        add = makexml("additional", "http://sumo.dlr.de/xsd/additional_file.xsd")
        for (rt, edge) in self.rts.items():
            add.append(E("route", id="route%s" % rt, edges=edge))
        printxml(add, self.cfg_path + addfn)

        gui = E("viewsettings")
        gui.append(E("scheme", name="real world"))
        printxml(gui, self.cfg_path +guifn)

        cfg = makexml("configuration", "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")

        logging.debug(self.netfn)

        cfg.append(self.inputs(self.name, net=self.netfn, add=addfn, rou=self.roufn, gui=guifn))
        t = E("time")
        t.append(E("begin", value=repr(start_time)))
        if end_time:
            t.append(E("end", value=repr(end_time)))
        cfg.append(t)

        printxml(cfg, self.cfg_path + cfgfn)
        return cfgfn

    def available_route_choices(self):
        """
        Specifies a dict on routes vehicles can traverse in the network

        :return: routes that vehicles in the network can traverse given their starting node
                 (only necessarily needed during initialization; afterwards, different routes
                  can be specified)
        """
        rts = {"AC":  "AC CB B BA1 BA2",
               "AD":  "AD DB B BA1 BA2",
               "CB":  "CB B BA1 BA2 AC",
               "CD":  "CD DB B BA1 BA2 AC",
               "DB":  "DB B BA1 BA2 AC CD",
               "B":   "B BA1 BA2",
               "BA1": "BA1 BA2",
               "BA2": "BA2"}

        return rts

    def make_routes(self, scenario, initial_config, cfg_params):

        type_params = scenario.type_params
        type_list = scenario.type_params.keys()
        num_cars = scenario.num_vehicles
        if type_list is not None:
            routes = makexml("routes", "http://sumo.dlr.de/xsd/routes_file.xsd")
            for tp in type_list:
                print(type_params[tp][1][0])
                if type_params[tp][1][0] == SumoController:
                    sumo_attributes = dict()
                    sumo_attributes["id"] = tp
                    sumo_attributes["minGap"] = "0"
                    for key in type_params[tp][1][1].keys():
                        sumo_attributes[key] = repr(type_params[tp][1][1][key])
                    routes.append(E("vType", attrib=sumo_attributes))
                else:
                    routes.append(E("vType", id=tp, minGap="0"))

            vehicle_ids = []
            if num_cars > 0:
                for type in type_list:
                    type_count = type_params[type][0]
                    for i in range(type_count):
                        vehicle_ids.append((type, type + "_" + str(i)))

            if initial_config["shuffle"]:
                random.shuffle(vehicle_ids)

            positions = initial_config["positions"]
            lanes = initial_config["lanes"]
            for i, (type, id) in enumerate(vehicle_ids):
                route, pos = positions[i]
                lane = lanes[i]
                type_depart_speed = type_params[type][3]
                routes.append(self.vehicle(type, "route" + route, depart="0", id=id, color="1,0.0,0.0",
                              departSpeed=str(type_depart_speed), departPos=str(pos), departLane=str(lane)))

            printxml(routes, self.cfg_path + self.roufn)
