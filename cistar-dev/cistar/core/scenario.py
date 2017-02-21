
import logging
from cistar.core.generator import Generator


class Scenario:

    def __init__(self, name, type_params, net_params, cfg_params=None, initial_config=None, cfg=None, generator_class=None):
        """
        Initializes a new scenario. This class can be instantiated once and reused in multiple experiments.

        :param name: A tag associated with the scenario
        :param type_params: dictionary of car -> (count, controller) assignments, where controller is a method
        :param net_params: Dict for parameters to be passed to the generator to be used when creating the config files
            i.e. numlanes, length, max_speed etc.
        :param cfg_params: Dict for parameters to be passed to the generator to be used when creating the config files
            i.e. start_time, end_time, cfg_path etc.
        :param initial_config:
        :param cfg:
        :param generator_class:
        """
        self.name = name
        self.num_vehicles = sum([x[1][0] for x in type_params.items()])
        self.type_params = type_params

        self.num_rl_vehicles = 0
        if "rl" in type_params:
            self.num_rl_vehicles = type_params["rl"][0]

        if not net_params:
            ValueError("No network params specified!")
        self.net_params = net_params

        if cfg:
            self.cfg = cfg
        elif not cfg:
            if not generator_class:
                ValueError("Must supply either a CFG or a simulator!!")
            self.generator_class = generator_class
            if cfg_params is None:
                ValueError("No config params specified")
            self.cfg_params = cfg_params

        self.initial_config = {}
        if initial_config:
            self.initial_config = initial_config

    def generate(self):
        logging.info("Config file not defined, generating using generator")

        net_path = Generator.NET_PATH
        cfg_path = Generator.CFG_PATH
        data_prefix = Generator.DATA_PREFIX

        if "net_path" in self.net_params:
            net_path = self.net_params["net_path"]
        if "cfg_path" in self.cfg_params:
            cfg_path = self.cfg_params["cfg_path"]
        if "data_prefix" in self.cfg_params:
            data_prefix = self.cfg_params["data_prefix"]

        generator = self.generator_class(net_path, cfg_path, data_prefix, self.name)
        generator.generate_net(self.net_params)
        cfg_name, outs = generator.generate_cfg(self.cfg_params)
        generator.makeRoutes(self, self.initial_config, self.cfg_params)
        return generator.cfg_path + cfg_name, outs

    def __str__(self):
        return "Scenario " + self.name + " with " + str(self.num_vehicles) + " vehicles."
