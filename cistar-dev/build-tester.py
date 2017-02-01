import logging

from cistar.envs.velocity import SimpleVelocityEnvironment
from cistar.core.exp import SumoExperiment
from rllab.envs.base import Env

from cistar.generators.loop import CircleGenerator

logging.basicConfig(level=logging.WARNING)

num_cars = 4

num_rl = 4

sumo_params = {"port": 8873}

sumo_binary = "sumo-gui"

type_controllers = {"rl": None}

type_counts = {"rl": num_rl}

env_params = {"target_velocity": 25}

initial_config = {}

net_params = {"length": 200, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"debug/net/"}

type_list=["rl"]

##data path needs to be relative to cfg location
cfg_params = {"type_list": ["rl"], "start_time": 0, "num_cars":num_cars, "end_time":3000, "cfg_path":"debug/cfg/", "type_counts":type_counts, "use_flows":True, "period":"1"}

leah_sumo_params = {"port": 8873}#, "cfg":"/Users/kanaad/code/research/learning-traffic/sumo/learning-traffic/cistar-dev/debug/cfg/test-exp-200m1l.sumo.cfg"}

exp = SumoExperiment("test-exp", SimpleVelocityEnvironment, env_params, num_cars, num_rl, type_controllers, sumo_binary, leah_sumo_params, initial_config, CircleGenerator, net_params, cfg_params)

logging.info("Experiment Set Up complete")

for _ in range(50):
    exp.env.step([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25])
exp.env.reset()
for _ in range(20):
    exp.env.step([15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])
for _ in range(10):
    exp.env.step([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25])
    
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

exp.env.terminate()