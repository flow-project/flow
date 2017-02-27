import os
import errno

from lxml import etree
E = etree.Element


def makexml(name, nsl):
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    ns = {"xsi": xsi}
    attr = {"{%s}noNamespaceSchemaLocation" % xsi: nsl}
    t = E(name, attrib=attr, nsmap=ns)
    return t


def printxml(t, fn):
    etree.ElementTree(t).write(fn, pretty_print=True, encoding='UTF-8', xml_declaration=True)

def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path



"""
Generator base class for creating 
"""
class Generator:

    CFG_PATH = "./"
    NET_PATH = "./"
    DATA_PREFIX = "data/"

    def __init__(self, net_path, cfg_path, data_prefix, base):
        self.net_path = net_path
        self.cfg_path = cfg_path
        self.data_prefix = data_prefix
        self.base = base
        self.name = base
        self.netfn = ""

        ensure_dir("%s" % self.net_path)
        ensure_dir("%s" % self.cfg_path)
        ensure_dir("%s" % self.cfg_path + self.data_prefix)

    def generate_net(self, net_params):
        raise NotImplementedError

    def generate_cfg(self, cfg_params):
        raise NotImplementedError

    def make_routes(self, scenario, initial_config, cfg_params):
        raise NotImplementedError


    def vtype(self, name, maxSpeed=30, accel=1.5, decel=4.5, length=5, **kwargs):
        return E("vType", accel=repr(accel), decel=repr(decel), id=name, length=repr(length),
                 maxSpeed=repr(maxSpeed), **kwargs)

    def flow(self, name, number, vtype, route, **kwargs):
        return E("flow", id=name, number=repr(number), route=route, type=vtype, **kwargs)

    def vehicle(self, type, route, departPos, number=0, id=None, **kwargs):
        if not id and not number:
            raise ValueError("Supply either ID or Number")
        if not id:
            id = type + "_" + str(number)
        return E("vehicle", type=type, id=id, route=route, departPos=departPos, **kwargs)

    def inputs(self, name, net=None, rou=None, add=None, gui=None):
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

    def outputs(self, name):
        t = E("output")
        outs = {"netstate": "dump",
                "amitran": "output",
                "lanechange": "output",
                "emission": "output", }

        for (key, val) in outs.items():
            fn = self.data_prefix + "%s.%s.xml" % (name, key)
            t.append(E("%s-%s" % (key, val), value=fn))
            outs[key] = fn
        return t, outs
