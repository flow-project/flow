from cistar.core.exp import Generator

from cistar.core.util import makexml
from cistar.core.util import printxml

import subprocess
import sys

from numpy import pi, sin, cos, linspace

import logging
import random
from lxml import etree
E = etree.Element


"""
Generator for intersections.
"""
class IntersectionGenerator(Generator):

    """
    Generates Net files for intersection sim. Requires:
    - length_top: length of the top portion of the intersection, in meters
    - length_bottom: length of the bottom portion of the intersection, in meters
    - length_left: length of the left portion of the intersection, in meters
    - length_right: length of the right portion of the intersection, in meters
    - lanes_top: number of lanes of the top section
    - lanes_bottom: number of lanes of the bottom section
    - lanes_left: number of lanes of the left section
    - lanes_right: number of lanes of the right section
    - speed_limit_top: max speed limit of the top section, in m/s
    - speed_limit_bottom: max speed limit of the bottom section, in m/s
    - speed_limit_left: max speed limit of the left section, in m/s
    - speed_limit_right: max speed limit of the right section, in m/s
    - direction_top_bottom: direction vehicles move on the vertical section (may be "up" or down")
    - direction_left_right: direction vehicles move on the horizontal section (may be "left" or "right")
    - angle: clockwise rotation of the intersection, in radians (default=0)
    - priority: specifies which portion of the intersection receives priority; can be "top_bottom", "left_right",
                or "None" (default="None"). If no priority is specified, vehicles may crash at the intersection.
    - resolution: number of nodes resolution
    """
    def generate_net(self, params):
        length_top = params["length_top"]
        length_bottom = params["length_top"]
        length_left = params["length_top"]
        length_right = params["length_top"]
        lanes_top = params["lanes_top"]
        lanes_bottom = params["lanes_bottom"]
        lanes_left = params["lanes_left"]
        lanes_right = params["lanes_right"]
        speed_limit_top = params["speed_limit_top"]
        speed_limit_bottom = params["speed_limit_bottom"]
        speed_limit_left = params["speed_limit_left"]
        speed_limit_right = params["speed_limit_right"]

        # TODO: what if we want vehicle to move in both directions (for different lanes)?
        # direction the vehicles will move on the vertical section
        if params["direction_top_bottom"] != "up" and params["direction_top_bottom"] != "down":
            # print error message
            angle = 1
        else:
            direction_top_bottom = params["direction_top_bottom"]

        # direction the vehicles will move on the horizontal section
        if params["direction_left_right"] != "left" and params["direction_left_right"] != "right":
            # print error message
            angle = 1
        else:
            direction_left_right = params["direction_left_right"]

        # the intersection will be rotated by this angle
        if not params["angle"]:
            angle = 0
        elif params["angle"] >= pi/2:
            # print warning
            angle = 0
        else:
            angle = params["angle"]

        # vehicles on sections with a lower priority value are given priority in crossing
        if not params["priority"]:
            priority_top_bottom = 1
            priority_left_right = 1
        elif params["priority"] == "None":
            priority_top_bottom = 1
            priority_left_right = 1
        elif params["priority"] == "top_bottom":
            priority_top_bottom = 1
            priority_left_right = 2
        elif params["priority"] == "left_right":
            priority_top_bottom = 2
            priority_left_right = 1

        resolution = params["resolution"]

        self.name = "%s-%dm,%dl_%dm,%dl_%dm,%dl_%dm,%dl" % (self.base, length_top, lanes_top, length_bottom,
                                                            lanes_bottom, length_left, lanes_left, length_right,
                                                            lanes_right)

        nodfn = "%s.nod.xml" % self.name
        edgfn = "%s.edg.xml" % self.name
        typfn = "%s.typ.xml" % self.name
        cfgfn = "%s.netccfg" % self.name
        netfn = "%s.net.xml" % self.name

        # xml file for nodes
        # contains nodes for the boundary points, as well as the center point
        # with respect to the x and y axes
        # titled: center, top, bottom, left, right
        x = makexml("nodes", "http://sumo.dlr.de/xsd/nodes_file.xsd")
        x.append(E("node", id="center", x=repr(0), y=repr(0)))
        x.append(E("node", id="top", x=repr(-length_top*sin(angle)), y=repr(length_top*cos(angle))))
        x.append(E("node", id="bottom", x=repr(length_bottom*sin(angle)), y=repr(-length_bottom*cos(angle))))
        x.append(E("node", id="left", x=repr(-length_left*cos(angle)), y=repr(-length_left*sin(angle))))
        x.append(E("node", id="right", x=repr(length_right*cos(angle)), y=repr(length_right*sin(angle))))
        printxml(x, self.net_path + nodfn)

        # xml file for edges
        # creates circular arcs that connect the created nodes
        # space between points in the edge is defined by the "resolution" variable
        x = makexml("edges", "http://sumo.dlr.de/xsd/edges_file.xsd")
        if direction_top_bottom == "up":
            from_top_bottom = "bottom"
            to_top_bottom = "top"
        elif direction_top_bottom == "down":
            from_top_bottom = "top"
            to_top_bottom = "bottom"

        if direction_left_right == "left":
            from_left_right = "right"
            to_left_right = "left"
        elif direction_left_right == "right":
            from_left_right = "left"
            to_left_right = "right"

        x.append(E("edge", attrib={"id": "top_bottom", "from": from_top_bottom, "to": to_top_bottom,
                                   "type": "edgeType", "length": repr(length_top+length_bottom)}))
        x.append(E("edge", attrib={"id": "left_right", "from": from_left_right, "to": to_left_right,
                                   "type": "edgeType", "length": repr(length_left+length_right)}))
        printxml(x, self.net_path + edgfn)

        # xml file for types
        # contains the the number of lanes and the speed limit for the lanes
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

        # netconvert -c $(cfg) --output-file=$(net)
        retcode = subprocess.call(
            ["netconvert -c " + self.net_path + cfgfn + " --output-file=" + self.cfg_path + netfn],
            stdout=sys.stdout, stderr=sys.stderr, shell=True)
        self.netfn = netfn

        return self.net_path + netfn

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
    def generate_cfg(self, params):

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
            i = E("interval", begin="0", end="100000")
            i.append(E("routeProbReroute", id=to))
            t.append(i)
            return t

        self.rts = {"top": "top left bottom right",
               "left": "left bottom right top",
               "bottom": "bottom right top left",
               "right": "right top left bottom"}

        add = makexml("additional", "http://sumo.dlr.de/xsd/additional_file.xsd")
        for (rt, edge) in self.rts.items():
            add.append(E("route", id="route%s" % rt, edges=edge))
        add.append(rerouter("rerouterBottom", "bottom", "routebottom"))
        add.append(rerouter("rerouterTop", "top", "routetop"))
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
        if type_list:
            routes = makexml("routes", "http://sumo.dlr.de/xsd/routes_file.xsd")
            for tp in type_list:
                routes.append(E("vType", id=tp, minGap="0"))

            vehicle_ids = []
            if num_cars > 0:
                for type in type_params:
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
