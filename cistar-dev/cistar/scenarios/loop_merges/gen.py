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


class LoopMergesGenerator(Generator):

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
        merge_in_len = params["merge_in_length"]
        merge_out_len = params["merge_out_length"]
        merge_in_angle = params["merge_in_angle"]
        merge_out_angle = params["merge_out_angle"]
        ring_radius = params["ring_radius"]

        lanes = params["lanes"]
        speed_limit = params["speed_limit"]
        res = params["resolution"]

        r = ring_radius

        length = 2 * pi * ring_radius + merge_in_len + merge_out_len

        self.name = "%s-%dm%dl" % (self.base, length, lanes)

        nodfn = "%s.nod.xml" % self.name
        edgfn = "%s.edg.xml" % self.name
        typfn = "%s.typ.xml" % self.name
        cfgfn = "%s.netccfg" % self.name
        netfn = "%s.net.xml" % self.name

        # xml file for nodes; contains nodes for the boundary points with respect to the x and y axes
        x = makexml("nodes", "http://sumo.dlr.de/xsd/nodes_file.xsd")

        x.append(E("node", id="merge_in", x=repr((r + merge_in_len) * cos(merge_in_angle)),
                   y=repr((r + merge_in_len) * sin(merge_in_angle)),  type="priority"))

        x.append(E("node", id="merge_out", x=repr((r + merge_out_len) * cos(merge_out_angle)),
                   y=repr((r + merge_out_len) * sin(merge_out_angle)), type="priority"))

        x.append(E("node", id="ring_0", x=repr(r * cos(merge_in_angle)),
                   y=repr(r * sin(merge_in_angle)), type="priority"))

        x.append(E("node", id="ring_1", x=repr(r * cos(merge_out_angle)),
                   y=repr(r * sin(merge_out_angle)), type="priority"))

        printxml(x, self.net_path + nodfn)

        # xml file for edges; creates circular arcs that connect the created nodes space between points
        # in the edge is defined by the "resolution" variable
        x = makexml("edges", "http://sumo.dlr.de/xsd/edges_file.xsd")

        # edges associated with merges
        x.append(E("edge", attrib={"id": "merge_in", "from": "merge_in", "to": "ring_0", "type": "edgeType",
                                   "length": repr(merge_in_len)}))
        x.append(E("edge", attrib={"id": "merge_out", "from": "ring_1", "to": "merge_out", "type": "edgeType",
                                   "length": repr(merge_out_len)}))

        # edges associated with the ring
        x.append(E("edge", attrib={"id": "ring_0", "from": "ring_0", "to": "ring_1", "type": "edgeType",
                                   "length": repr((merge_out_angle - merge_in_angle) % (2 * pi) * r),
                                   "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                                      for t in linspace(merge_in_angle, merge_out_angle, res)])}))
        x.append(E("edge", attrib={"id": "ring_1", "from": "ring_1", "to": "ring_0", "type": "edgeType",
                                   "length": repr((merge_in_angle - merge_out_angle) % (2 * pi) * r),
                                   "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                                      for t in linspace(merge_out_angle, 2 * pi + merge_in_angle, res)])}))

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

        self.rts = {"ring_0":   "ring_0 ring_1",
                    "ring_1":   "ring_1 ring_0",
                    "merge_in": "merge_in ring_0 merge_out"}

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
            ring_positions = positions[:scenario.num_vehicles-scenario.num_merge_vehicles]
            merge_positions = positions[scenario.num_vehicles-scenario.num_merge_vehicles:]
            i_merge = 0
            i_ring = 0
            for i, (type, id) in enumerate(vehicle_ids):
                if "merge" in type:
                    route, pos = merge_positions[i_merge]
                    i_merge += 1
                else:
                    route, pos = ring_positions[i_ring]
                    i_ring += 1

                type_depart_speed = type_params[type][3]
                routes.append(self.vehicle(type, "route" + route, depart="0",
                              departSpeed=str(type_depart_speed), departPos=str(pos), id=id, color="1,0.0,0.0"))

            printxml(routes, self.cfg_path + self.roufn)
