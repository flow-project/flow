from EnvironmentBase import SumoEnvironment
from SumoExperiment import SumoExperiment
from rllab.envs.base import Env

# from CircleGenerator import CircleGenerator

from LoopExperiment import SimpleVelocityEnvironment

import logging

class test():
    def __init__(self):
        self.dict = {"oh": "fuck"}

logging.basicConfig(level=logging.INFO)

sumo_params = {"port": 8873}

sumo_binary = "sumo"

vehicle_controllers = {"rl": (2, None)}

env_params = {"target_velocity": 25}

initial_config = {}

net_params = {"length": 200, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"debug/net/"}

##data path needs to be relative to cfg location
cfg_params = {"type_list": ["rl"], "start_time": 0, "end_time":3000, "cfg_path":"debug/cfg/"}

#exp = SumoExperiment("test-exp", SumoEnvironment, env_params, 1, vehicle_controllers, sumo_binary, sumo_params, initial_config, CircleGenerator, net_params, cfg_params)

leah_sumo_params = {"port": 8873, "cfg":"/Users/kanaad/code/research/learning-traffic/sumo/learning-traffic/cistar/debug/cfg/test-exp-200m1l.sumo.cfg"}

exp = SumoExperiment("test-exp", SumoEnvironment, env_params, 2, vehicle_controllers, sumo_binary, leah_sumo_params, initial_config)#, CircleGenerator, net_params, cfg_params)

logging.info("Experiment Set Up complete")

import json

with open("out.json", "w") as test:
    json.dump(Env, test)
    print("dumps")


#
# for _ in range(20):
#     exp.env.step([25, 25, 25, 25])
# print("resetting state")
# exp.env.reset()
# print("state reset")
# for _ in range(20):
#     exp.env.step([0,0,0,0])
# print("resetting state")
# exp.env.reset()
# print("state reset")
# for _ in range(10):
#     exp.env.step([20,20,20,20])
# for _ in range(30):
#     exp.env.step([25, 25, 25, 25])
# print("resetting state")
# exp.env.reset()
