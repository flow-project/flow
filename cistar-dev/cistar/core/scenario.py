
import logging
from cistar.core.generator import Generator


class Scenario:

    def __init__(self, name, num_vehicles, type_params, cfg_params, net_params, initial_params=None, cfg=None, generator_class=None):
        self.num_vehicles = num_vehicles
        self.type_params = type_params
        self.num_rl_vehicles = 0
        if "rl" in type_params:
            self.num_rl_vehicles = type_params["rl"][0]
        self.cfg_params = cfg_params
        self.net_params = net_params
        self.name = name

        if not initial_params:
            self.initial_config = {}
        else:
            self.initial_config = initial_params

        if net_params is None:
            logging.error("No network params specified!")
        if cfg_params is None:
            logging.error("No config params specified")

        if cfg:
            self.cfg = cfg
        elif not cfg and not generator_class:
            ValueError("Must supply either a CFG or a simulator!!")
        elif generator_class:
            self.generator_class = generator_class


    def generate(self):
        logging.info(" Config file not defined, generating using generator")

        net_path = Generator.NET_PATH
        cfg_path = Generator.CFG_PATH
        data_prefix = Generator.DATA_PREFIX

        if "net_path" in self.net_params:
            net_path = self.net_params["net_path"]
        if "cfg_path" in self.cfg_params:
            cfg_path = self.cfg_params["cfg_path"]
        if "data_prefix" in self.cfg_params:
            data_prefix = self.cfg_params["data_prefix"]

        self.generator = self.generator_class(net_path, cfg_path, data_prefix, self.name)
        self.generator.generate_net(self.net_params)
        cfg_name, self.outs = self.generator.generate_cfg(self.cfg_params)
        self.generator.makeRoutes(self, self.initial_config, self.cfg_params)
        return self.generator.cfg_path + cfg_name
