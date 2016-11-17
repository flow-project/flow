import subprocess
import sys
from lxml import etree
from numpy import pi, sin, cos, linspace
import SumoExperiment

E = etree.Element

DATA_PREFIX = "data/"


def makexml(name, nsl):
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    ns = {"xsi": xsi}
    attr = {"{%s}noNamespaceSchemaLocation" % xsi: nsl}
    t = E(name, attrib=attr, nsmap=ns)
    return t


def printxml(t, fn):
    etree.ElementTree(t).write(fn, pretty_print=True, encoding='UTF-8', xml_declaration=True)

"""
Generator for loop circle used in MIT traffic simulation.
"""
class CircleGenerator(SumoExperiment.Generator):


    """
    Generates Net files for loop sim. Requires:
    length: length of the circle
    lanes: number of lanes in the circle
    speed_limit: max speed limit of the circle
    resolution: number of nodes resolution

    """
    def generatenet(self, params):

        if "length" not in params:
            raise ValueError("length of circle not supplied")
        else:
            length = params["length"]

        if "lanes" not in params:
            raise ValueError("lanes of circle not supplied")
        else:
            lanes = params["lanes"]

        if "speed_limit" not in params:
            raise ValueError("speed limit of circle not supplied")
        else:
            speed_limit = params["speed_limit"]

        if "resolution" not in params:
            raise ValueError("speed limit of circle not supplied")
        else:
            resolution = params["speed_limit"]

        self.name = "%s-%dm%dl" % (self.base, length, lanes)

        nodfn = "%s.nod.xml" % self.name
        edgfn = "%s.edg.xml" % self.name
        typfn = "%s.typ.xml" % self.name
        cfgfn = "%s.netccfg" % self.name
        netfn = "%s.net.xml" % self.name

        r = length / pi
        edgelen = length / 4.

        x = makexml("nodes", "http://sumo.dlr.de/xsd/nodes_file.xsd")
        x.append(E("node", id="bottom", x=repr(0), y=repr(-r)))
        x.append(E("node", id="right", x=repr(r), y=repr(0)))
        x.append(E("node", id="top", x=repr(0), y=repr(r)))
        x.append(E("node", id="left", x=repr(-r), y=repr(0)))
        printxml(x, self.path + nodfn)

        x = makexml("edges", "http://sumo.dlr.de/xsd/edges_file.xsd")
        x.append(E("edge", attrib={"id": "bottom", "from": "bottom", "to": "right", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                                      for t in linspace(-pi / 2, 0, resolution)]),
                                   "length": repr(edgelen)}))
        x.append(E("edge", attrib={"id": "right", "from": "right", "to": "top", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                                      for t in linspace(0, pi / 2, resolution)]),
                                   "length": repr(edgelen)}))
        x.append(E("edge", attrib={"id": "top", "from": "top", "to": "left", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                                      for t in linspace(pi / 2, pi, resolution)]),
                                   "length": repr(edgelen)}))
        x.append(E("edge", attrib={"id": "left", "from": "left", "to": "bottom", "type": "edgeType",
                                   "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                                      for t in linspace(pi, 3 * pi / 2, resolution)]),
                                   "length": repr(edgelen)}))
        printxml(x, self.path + edgfn)

        x = makexml("types", "http://sumo.dlr.de/xsd/types_file.xsd")
        x.append(E("type", id="edgeType", numLanes=repr(lanes), speed=repr(speed_limit)))
        printxml(x, self.path + typfn)

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
        printxml(x, self.path + cfgfn)

        # netconvert -c $(cfg) --output-file=$(net)
        retcode = subprocess.call(
            ['netconvert', "-c", self.path + cfgfn],
            stdout=sys.stdout, stderr=sys.stderr, shell=True)

        self.netfn = netfn

        return self.path + netfn


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
    def generatecfg(self, params):

        if "num_cars" not in params:
            if "type_list" not in params:
                raise ValueError("type_list or num_cars of circle not supplied")
            else:
                type_list = params["type_list"]
        else:
            num_cars = params["num_cars"]

            if "max_speed" not in params:
                raise ValueError("max_speed of circle not supplied")
            else:
                max_speed = params["max_speed"]



        if "start_time" not in params:
            raise ValueError("start_time of circle not supplied")
        else:
            start_time = params["start_time"]

        if "end_time" not in params:
            raise ValueError("end_time of circle not supplied")
        else:
            end_time = params["end_time"]

        roufn = "%s.rou.xml" % self.name
        addfn = "%s.add.xml" % self.name
        cfgfn = "%s.sumo.cfg" % self.name
        guifn = "%s.gui.cfg" % self.name

        def rerouter(name, frm, to):
            t = E("rerouter", id=name, edges=frm)
            i = E("interval", begin="0", end="100000")
            i.append(E("routeProbReroute", id=to))
            t.append(i)
            return t

        def vtype(name, maxSpeed=30, accel=1.5, decel=4.5, length=5, **kwargs):
            return E("vType", accel=repr(accel), decel=repr(decel), id=name, length=repr(length),
                     maxSpeed=repr(maxSpeed), **kwargs)

        def flow(name, number, vtype, route, **kwargs):
            return E("flow", id=name, number=repr(number), route=route, type=vtype, **kwargs)

        def inputs(name, net=None, rou=None, add=None, gui=None):
            inp = E("input")
            if net is not False:
                if net is None:
                    inp.append(E("net-file", value="%s.net.xml" % name))
                else:
                    inp.append(E("net-file", value=net))
            if rou is not False:
                if rou is None:
                    inp.append(E("route-files", value="%s.rou.xml" % name))
                else:
                    inp.append(E("route-files", value=rou))
            if add is not False:
                if add is None:
                    inp.append(E("additional-files", value="%s.add.xml" % name))
                else:
                    inp.append(E("additional-files", value=add))
            if gui is not False:
                if gui is None:
                    inp.append(E("gui-settings-file", value="%s.gui.xml" % name))
                else:
                    inp.append(E("gui-settings-file", value=gui))
            return inp

        def outputs(name, prefix="data/"):
            t = E("output")
            outs = {"netstate": "dump",
                    "amitran": "output",
                    "lanechange": "output",
                    "emission": "output", }

            for (key, val) in outs.iteritems():
                fn = prefix + "%s.%s.xml" % (name, key)
                t.append(E("%s-%s" % (key, val), value=fn))
                outs[key] = fn
            return t, outs

        rts = {"top": "top left bottom right",
               "left": "left bottom right top",
               "bottom": "bottom right top left",
               "right": "right top left bottom"}

        add = makexml("additional", "http://sumo.dlr.de/xsd/additional_file.xsd")
        for (rt, edge) in rts.items():
            add.append(E("route", id="route%s" % rt, edges=edge))
        add.append(rerouter("rerouterBottom", "bottom", "routebottom"))
        add.append(rerouter("rerouterTop", "top", "routetop"))
        printxml(add, addfn)

        if num_cars > 0:
            routes = makexml("routes", "http://sumo.dlr.de/xsd/routes_file.xsd")
            routes.append(vtype("car", max_speed))
            for rt in rts:
                routes.append(flow("car%s" % rt, num_cars / len(rts), "car", "route%s" % rt,
                                   begin="0", period="1", departPos="free"))
            printxml(routes, roufn)
        elif type_list:
            routes = makexml("routes", "http://sumo.dlr.de/xsd/routes_file.xsd")
            for tp in type_list:
                routes.append(E("vType", id=tp))
            printxml(routes, roufn)
        else:
            roufn = False

        gui = E("viewsettings")
        gui.append(E("scheme", name="real world"))
        printxml(gui, guifn)

        cfg = makexml("configuration", "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
        cfg.append(inputs(self.name, net=self.netfn, add=addfn, rou=roufn, gui=guifn))
        t, outs = outputs(self.name, prefix=DATA_PREFIX)
        cfg.append(t)
        t = E("time")
        t.append(E("begin", value=repr(start_time)))
        t.append(E("end", value=repr(end_time)))
        cfg.append(t)

        printxml(cfg, cfgfn)
        return cfgfn, outs
