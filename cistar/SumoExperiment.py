import logging
import datetime

import subprocess
import sys
import os
import errno

"""
Primary sumo++ file, imports API from supporting files and manages in teractions
with rllab and custom controllers.

In addition to opening a traci port and running an instance of Sumo, the 
simulation class should store the controllers for both the manual and the 
autonomous vehicles, which it will use implement actions on the vehicles.

Interfaces with sumo on the other side

"""


def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


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

    def generate_net(self, params):
        raise NotImplementedError

    def generate_cfg(self, params):
        raise NotImplementedError


class SumoExperiment():

    def __init__(self, name, env_class, env_params, num_rl_vehicles, vehicle_params, sumo_binary, sumo_params, initial_config, file_generator=None, net_params=None,
                 cfg_params=None):
        """
        name : tag to associate experiment with
        env_class : Environment to be initialized
        num_rl_vehicles :  these vehicles do not have a controller assigned to them
        vehicle_params : vehicle type string -> (number of vehicle, controller)
            example:
             num_rl_vehicles=8
             vehicle params:
               'acc' -> (5, acc_controller_fn)
               'human-0.5-delay' -> (10, human-delay)
               'human-1.0-delay' -> (10, human-delay)
               'rl' -> (8, None)

        sumo_binary : path to sumo executable
        sumo_params : parameters to pass to sumo, e.g. step-length (can also be in sumo-cfg)
        file_generator : Child of Generator that will create the net, cfg files
        net_params : Dict for parameters for netgenerator
            i.e. for loops, includes, numlanes, length
        vehicles : Dict mapping type of vehicle to tuple with (number of instances of the vehicle, controller for vehicle type)
        cfg_params : params to be passed to the environement class upon initialization
        cfg : specify a configuration, rather than create a new one
        """
        self.name = name
        self.vehicle_controllers = {}
        self.num_rl_vehicles = num_rl_vehicles
        self.env_params = env_params
        self.initial_config = initial_config

        logging.info(" Starting experiment" + str(name) + " at " + str(datetime.datetime.utcnow()))

        total_rl_in_params = 0
        for vehicle_type in vehicle_params:
            num_instances, controller = vehicle_params[vehicle_type]
            if controller is None:
                total_rl_in_params += num_instances
            for i in range(num_instances):
                veh_id = vehicle_type + "_" + str(i)
                self.vehicle_controllers[veh_id] = controller

        if total_rl_in_params != num_rl_vehicles:
            logging.error("Error! Invalid number of rl_vehicles. Are you miscounting or is a controller not specified?")

        self.num_vehicles = len(self.vehicle_controllers)
        if "cfg" not in sumo_params:
            logging.info(" Config file not defined, generating using generator")
            if file_generator is None:
                logging.error("Invalid file generator!")
            elif net_params is None:
                logging.error("No network params specifed!")
            elif cfg_params is None:
                logging.error("No config params specified")
            else:
                net_path = Generator.NET_PATH
                cfg_path = Generator.CFG_PATH
                data_prefix = Generator.DATA_PREFIX

                if "net_path" in net_params:
                    net_path = net_params["net_path"]
                if "cfg_path" in cfg_params:
                    cfg_path = cfg_params["cfg_path"]
                if "data_prefix" in cfg_params:
                    data_prefix = cfg_params["data_prefix"]

                generator = file_generator(net_path, cfg_path, data_prefix, self.name)
                generator.generate_net(net_params)
                self.cfg, self.outs = generator.generate_cfg(cfg_params)
                sumo_params['cfg'] = generator.cfg_path + self.cfg

        logging.info(" initializing enviornment.")

        self.env = env_class(self.num_vehicles, self.num_rl_vehicles, self.env_params, self.vehicle_controllers, sumo_binary, sumo_params, initial_config)

    def getCfg(self):
        return self.cfg

    def getEnv(self):
        return self.env
