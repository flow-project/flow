import subprocess
import sys
import os
import errno

"""
Primary sumo++ file, imports API from supporting files and manages interactions 
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
    def __init__(self, net_path, data_path, base):
        self.net_path = net_path
        self.data_path = data_path
        self.base = base
        self.name = base
        self.netfn = ""

        self.net_path = ensure_dir("%s" % self.net_path)
        self.data_path = ensure_dir("%s" % self.data_path)

    def generatenet(self, params):
        raise NotImplementedError

    def generatecfg(self, params):
        raise NotImplementedError


class SumoExperiment():
    """
    env: Environment to be initialized.
    netgenerator: Child of Generator that will create the net, cfg files
    netparams: Dict for parameters for netgenerator
        i.e. for loops, includes, numlanes, length
    vehicles: Dict mapping type of vehicle to tuple with (number of instances of the vehicle, controller for vehicle type)
    sumobinary: path to sumo executable
    sumoparams: parameters to pass to sumo, e.g. step-length (can also be in sumo-cfg)
    """

    def __init__(self, name, env_class, vehicle_params, sumo_binary, sumo_params, file_generator=None, net_params=None,
                 cfg_params=None, cfg=None):
        self.name = name
        self.vehicle_controllers = {}

        for vehicle_type in vehicle_params:
            num_instances = vehicle_params[vehicle_type][0]
            controller = vehicle_params[vehicle_type][1]
            for i in range(num_instances):
                veh_id = vehicle_type + "_" + str(i)
            self.vehicle_controllers[veh_id] = controller

        self.num_vehicles = len(self.vehicle_controllers)
        if not cfg:
            file_generator.generatenet(net_params)
            self.cfg, self.outs = file_generator.generatecfg(cfg_params)
        else:
            self.cfg = cfg

        self.env = env_class.__init__(self.num_vehicles, self.vehicle_controllers, sumo_binary, sumo_params)

    def getCfg(self):
        return self.cfg

    def getEnv(self):
        return self.env
