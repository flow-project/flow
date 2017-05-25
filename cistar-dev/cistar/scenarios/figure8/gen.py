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
Generator for figure 8 lanes.
"""
class Figure8Generator(Generator):

    """
    Generates Net files for loop sim. Requires:
    - radius_ring: radius of the ring portion of the network
    - lanes: number of lanes in the road network
    - priority: specifies which portion of the intersection receives priority; can be "top_bottom", "left_right",
                or "None" (default="None"). If no priority is specified, vehicles may crash at the intersection.
    - speed_limit: max speed limit of the vehicles on the road network
    - resolution: number of nodes resolution
    """
    def generate_net(self, params):
        r = params["radius_ring"]
        lanes = params["lanes"]
        speed_limit = params["speed_limit"]
        resolution = params["resolution"]

        # vehicles on sections with a lower priority value are given priority in crossing
        intersection_type = "unregulated"
        priority_top_bottom = 0
        priority_left_right = 0
        if not params["priority"]:
            pass
        elif params["priority"] == "top_bottom":
            intersection_type = "priority"
            priority_top_bottom = 46
            priority_left_right = 78
        elif params["priority"] == "left_right":
            intersection_type = "priority"
            priority_top_bottom = 78
            priority_left_right = 46

        ring_edgelen = r * pi/2.
        intersection_edgelen = 2*r

        self.name = "%s-%dm%dl" % (self.base, 2*intersection_edgelen+6*ring_edgelen, lanes)

        nodfn = "%s.nod.xml" % self.name
        edgfn = "%s.edg.xml" % self.name
        typfn = "%s.typ.xml" % self.name
        cfgfn = "%s.netccfg" % self.name
        netfn = "%s.net.xml" % self.name

        # xml file for nodes
        # contains nodes for the boundary points
        # with respect to the x and y axes
        # titled: center_intersection,
        #         top_lower_ring, bottom_lower_ring, left_lower_ring, right_lower_ring,
        #         top_upper_ring, bottom_upper_ring, left_upper_ring, right_upper_ring
        x = makexml("nodes", "http://sumo.dlr.de/xsd/nodes_file.xsd")
        x.append(E("node", id="center_intersection", x=repr(0), y=repr(0)))
        x.append(E("node", id="top_upper_ring", x=repr(r), y=repr(2*r), type=intersection_type))
        x.append(E("node", id="bottom_upper_ring", x=repr(r), y=repr(0), type=intersection_type))
        x.append(E("node", id="left_upper_ring", x=repr(0), y=repr(r), type=intersection_type))
        x.append(E("node", id="right_upper_ring", x=repr(2*r), y=repr(r), type=intersection_type))
        x.append(E("node", id="top_lower_ring", x=repr(-r), y=repr(0), type=intersection_type))
        x.append(E("node", id="bottom_lower_ring", x=repr(-r), y=repr(-2*r), type=intersection_type))
        x.append(E("node", id="left_lower_ring", x=repr(-2*r), y=repr(-r), type=intersection_type))
        x.append(E("node", id="right_lower_ring", x=repr(0), y=repr(-r), type=intersection_type))

        printxml(x, self.net_path + nodfn)

        # xml file for edges
        # creates circular arcs that connect the created nodes
        # space between points in the edge is defined by the "resolution" variable
        x = makexml("edges", "http://sumo.dlr.de/xsd/edges_file.xsd")

        # # intersection edges
        x.append(E("edge", attrib={"id": "right_lower_ring_in",  # "width": "5",
                                   "from": "right_lower_ring", "to": "center_intersection", "type": "edgeType",
                                   "length": repr(intersection_edgelen/2),
                                   "priority": repr(priority_top_bottom)}))
        x.append(E("edge", attrib={"id": "right_lower_ring_out",  # "width": "5",
                                   "from": "center_intersection", "to": "left_upper_ring", "type": "edgeType",
                                   "length": repr(intersection_edgelen/2),
                                   "priority": repr(priority_top_bottom)}))
        x.append(E("edge", attrib={"id": "bottom_upper_ring_in",  # "width": "5",
                                   "from": "bottom_upper_ring", "to": "center_intersection", "type": "edgeType",
                                   "length": repr(intersection_edgelen/2),
                                   "priority": repr(priority_left_right)}))
        x.append(E("edge", attrib={"id": "bottom_upper_ring_out",  # "width": "5",
                                   "from": "center_intersection", "to": "top_lower_ring", "type": "edgeType",
                                   "length": repr(intersection_edgelen/2),
                                   "priority": repr(priority_left_right)}))

        # ring edges
        x.append(E("edge", attrib={"id": "left_upper_ring",  # "width": "5",
                                   "from": "left_upper_ring", "to": "top_upper_ring", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (r * (1 - cos(t)), r * (1 + sin(t)))
                                                      for t in linspace(0, pi/2, resolution)]),
                                   "length": repr(ring_edgelen)}))
        x.append(E("edge", attrib={"id": "top_upper_ring",  # "width": "5",
                                   "from": "top_upper_ring", "to": "right_upper_ring", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (r * (1 + sin(t)), r * (1 + cos(t)))
                                                      for t in linspace(0, pi/2, resolution)]),
                                   "length": repr(ring_edgelen)}))
        x.append(E("edge", attrib={"id": "right_upper_ring",  # "width": "5",
                                   "from": "right_upper_ring", "to": "bottom_upper_ring", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (r * (1 + cos(t)), r * (1 - sin(t)))
                                                      for t in linspace(0, pi/2, resolution)]),
                                   "length": repr(ring_edgelen)}))
        x.append(E("edge", attrib={"id": "top_lower_ring",  # "width": "5",
                                   "from": "top_lower_ring", "to": "left_lower_ring", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (- r + r * cos(t), -r + r * sin(t))
                                                      for t in linspace(pi/2, pi, resolution)]),
                                   "length": repr(ring_edgelen)}))
        x.append(E("edge", attrib={"id": "left_lower_ring",  # "width": "5",
                                   "from": "left_lower_ring", "to": "bottom_lower_ring", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (- r + r * cos(t), - r + r * sin(t))
                                                      for t in linspace(pi, 3*pi/2, resolution)]),
                                   "length": repr(ring_edgelen)}))
        x.append(E("edge", attrib={"id": "bottom_lower_ring",  # "width": "5",
                                   "from": "bottom_lower_ring", "to": "right_lower_ring", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (- r + r * cos(t), - r + r * sin(t))
                                                      for t in linspace(-pi/2, 0, resolution)]),
                                   "length": repr(ring_edgelen)}))

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
        retcode = subprocess.call(["netconvert -c " + self.net_path + cfgfn + " --output-file=" +
                                   self.cfg_path + netfn + ' --no-internal-links="false"'],
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
            i = E("interval", begin="0", end="10000000")
            i.append(E("routeProbReroute", id=to))
            t.append(i)
            return t

        self.rts = {"bottom_lower_ring":     "bottom_lower_ring right_lower_ring_in right_lower_ring_out "
                                             "left_upper_ring top_upper_ring right_upper_ring bottom_upper_ring_in "
                                             "bottom_upper_ring_out top_lower_ring left_lower_ring",
                    "right_lower_ring_in":   "right_lower_ring_in right_lower_ring_out left_upper_ring top_upper_ring "
                                             "right_upper_ring bottom_upper_ring_in bottom_upper_ring_out "
                                             "top_lower_ring left_lower_ring bottom_lower_ring",
                    "right_lower_ring_out":  "right_lower_ring_out left_upper_ring top_upper_ring right_upper_ring "
                                             "bottom_upper_ring_in bottom_upper_ring_out top_lower_ring "
                                             "left_lower_ring bottom_lower_ring right_lower_ring_in",
                    "left_upper_ring":       "left_upper_ring top_upper_ring right_upper_ring bottom_upper_ring_in "
                                             "bottom_upper_ring_out top_lower_ring left_lower_ring bottom_lower_ring "
                                             "right_lower_ring_in right_lower_ring_out",
                    "top_upper_ring":        "top_upper_ring right_upper_ring bottom_upper_ring_in "
                                             "bottom_upper_ring_out top_lower_ring left_lower_ring bottom_lower_ring "
                                             "right_lower_ring_in right_lower_ring_out left_upper_ring",
                    "right_upper_ring":      "right_upper_ring bottom_upper_ring_in bottom_upper_ring_out "
                                             "top_lower_ring left_lower_ring bottom_lower_ring right_lower_ring_in "
                                             "right_lower_ring_out left_upper_ring top_upper_ring",
                    "bottom_upper_ring_in":  "bottom_upper_ring_in bottom_upper_ring_out top_lower_ring "
                                             "left_lower_ring bottom_lower_ring right_lower_ring_in "
                                             "right_lower_ring_out left_upper_ring top_upper_ring right_upper_ring",
                    "bottom_upper_ring_out": "bottom_upper_ring_out top_lower_ring left_lower_ring "
                                             "bottom_lower_ring right_lower_ring_in right_lower_ring_out "
                                             "left_upper_ring top_upper_ring right_upper_ring bottom_upper_ring_in",
                    "top_lower_ring":        "top_lower_ring left_lower_ring bottom_lower_ring right_lower_ring_in "
                                             "right_lower_ring_out left_upper_ring top_upper_ring "
                                             "right_upper_ring bottom_upper_ring_in bottom_upper_ring_out",
                    "left_lower_ring":       "left_lower_ring bottom_lower_ring right_lower_ring_in "
                                             "right_lower_ring_out left_upper_ring top_upper_ring right_upper_ring "
                                             "bottom_upper_ring_in bottom_upper_ring_out top_lower_ring"}

        add = makexml("additional", "http://sumo.dlr.de/xsd/additional_file.xsd")
        for (rt, edge) in self.rts.items():
            add.append(E("route", id="route%s" % rt, edges=edge))
        add.append(rerouter("rerouterBottom_lower_ring", "bottom_lower_ring", "routetop_upper_ring"))
        add.append(rerouter("rerouterLeft_upper_ring", "left_upper_ring", "routeright_lower_ring_in"))
        add.append(rerouter("rerouterTop_upper_ring", "top_upper_ring", "routebottom_lower_ring"))
        add.append(rerouter("rerouterRight_upper_ring", "right_upper_ring", "routeleft_lower_ring"))
        add.append(rerouter("rerouterTop_lower_ring", "top_lower_ring", "routetop_upper_ring"))
        add.append(rerouter("rerouterLeft_lower_ring", "left_lower_ring", "routeright_upper_ring"))
        printxml(add, self.cfg_path + addfn)

        gui = E("viewsettings")
        gui.append(E("scheme", name="real world"))
        printxml(gui, self.cfg_path + guifn)

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
