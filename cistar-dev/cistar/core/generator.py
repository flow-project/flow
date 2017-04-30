import errno
import os

from lxml import etree
from cistar.core.util import ensure_dir

from rllab.core.serializable import Serializable

E = etree.Element

"""
Generator base class for creating 
"""


class Generator(Serializable):
    CFG_PATH = "./"
    NET_PATH = "./"

    def __init__(self, net_path, cfg_path, base):
        Serializable.quick_init(self, locals())

        self.net_path = net_path
        self.cfg_path = cfg_path
        self.base = base
        self.name = base
        self.netfn = ""

        ensure_dir("%s" % self.net_path)
        ensure_dir("%s" % self.cfg_path)

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