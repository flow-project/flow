from cistar.core.exp import Generator
from cistar.controllers.base_controller import SumoController

from cistar.core.util import makexml
from cistar.core.util import printxml

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
        - horizontal_length_in: length of the horizontal lane before the intersection
        - horizontal_length_out: length of the horizontal lane after the intersection
        - horizontal_lanes: number of lanes in the horizontal lane
        - vertical_length_in: length of the vertical lane before the intersection
        - vertical_length_out: length of the vertical lane after the intersection
        - vertical_lanes: number of lanes in the vertical lane
        - speed_limit: max speed limit of the vehicles on the road network
        """
        edge_len = params["edge_length"]
        lanes = params["lanes"]
        speed_limit = params["speed_limit"]
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

        # xml file for nodes; contains nodes for the boundary points with respect to the x and y axes
        x = makexml("nodes", "http://sumo.dlr.de/xsd/nodes_file.xsd")
        x.append(E("node", id="A",   x=repr(0),          y=repr(0),       type="priority"))
        x.append(E("node", id="C",   x=repr(edge_x),     y=repr(edge_y),  type="priority"))
        x.append(E("node", id="D",   x=repr(edge_x),     y=repr(-edge_y), type="priority"))
        x.append(E("node", id="B",   x=repr(2 * edge_x), y=repr(0),       type="priority"))
        x.append(E("node", id="BA1", x=repr(2 * edge_x), y=repr(-2 * r),  type="priority"))
        x.append(E("node", id="BA2", x=repr(0),          y=repr(-2 * r),  type="priority"))
        printxml(x, self.net_path + nodfn)

        # xml file for edges; creates circular arcs that connect the created nodes space between points
        # in the edge is defined by the "resolution" variable

        # braess network component
        x = makexml("edges", "http://sumo.dlr.de/xsd/edges_file.xsd")
        x.append(E("edge", attrib={"id": "AC", "from": "A", "to": "C", "type": "edgeType",
                                   "length": repr(edge_len)}))
        x.append(E("edge", attrib={"id": "AD", "from": "A", "to": "D", "type": "edgeType",  # "priority": "78",
                                   "length": repr(edge_len)}))
        x.append(E("edge", attrib={"id": "CB", "from": "C", "to": "B", "type": "edgeType",
                                   "length": repr(edge_len)}))
        x.append(E("edge", attrib={"id": "CD", "from": "C", "to": "D", "type": "edgeType",  # "priority": "46",
                                   "length": repr(2 * edge_y)}))
        x.append(E("edge", attrib={"id": "D", "from": "D", "to": "B", "type": "edgeType",
                                   "length": repr(edge_len)}))

        # connecting output to input in braess network (to produce loop)
        x.append(E("edge", attrib={"id": "B", "from": "B", "to": "BA1", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (2 * edge_x + r * sin(t), r * (- 1 + cos(t)))
                                                      for t in linspace(0, pi, resolution)]),
                                   "length": repr(curve_len)}))
        x.append(E("edge", attrib={"id": "BA1", "from": "BA1", "to": "BA2", "type": "edgeType",
                                   "length": repr(straight_horz_len)}))
        x.append(E("edge", attrib={"id": "BA2", "from": "BA2", "to": "A", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (- r * sin(t), - r * (1 + cos(t)))
                                                      for t in linspace(0, pi, resolution)]),
                                   "length": repr(curve_len)}))

        printxml(x, self.net_path + edgfn)

        # xml file for types; contains the the number of lanes and the speed limit for the lanes
        x = makexml("types", "http://sumo.dlr.de/xsd/types_file.xsd")
        x.append(E("type", id="edgeType", numLanes=repr(lanes), speed=repr(speed_limit)))
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
        x.append(t)
        t = E("output")
        t.append(E("output-file", value=netfn))
        x.append(t)
        t = E("processing")
        t.append(E("no-internal-links", value="true"))
        t.append(E("no-turnarounds", value="true"))
        x.append(t)
        printxml(x, self.net_path + cfgfn)

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

        def rerouter(name, frm, to):
            '''

            :param name:
            :param frm:
            :param to:
            :return:
            '''
            t = E("rerouter", id=name, edges=frm)
            i = E("interval", begin="0", end="10000000")
            i.append(E("routeProbReroute", id=to))
            t.append(i)
            return t

        self.rts = {"AC":  "AC CB B BA1 BA2",
                    "AD":  "AD D B BA1 BA2",
                    "CB":  "CB B BA1 BA2 AC",
                    "CD":  "CD D B BA1 BA2 AC",
                    "D":   "D B BA1 BA2 AC CD",
                    "B":   "B BA1 BA2 AC CD D",
                    "BA1": "BA1 BA2 AC CB B",
                    "BA2": "BA2 AC CB B BA1"}

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
            for i, (type, id) in enumerate(vehicle_ids):
                route, pos = positions[i]
                type_depart_speed = type_params[type][3]
                routes.append(self.vehicle(type, "route" + route, depart="0",
                              departSpeed=str(type_depart_speed), departPos=str(pos), id=id, color="1,0.0,0.0"))

            printxml(routes, self.cfg_path + self.roufn)
