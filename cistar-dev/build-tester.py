import logging

from cistar.core.exp import SumoExperiment
from cistar.envs.velocity import SimpleVelocityEnvironment
from cistar.scenarios.loop.gen import CircleGenerator
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.car_following_models import *

logging.basicConfig(level=logging.WARNING)

num_cars = 10

sumo_params = {"port": 8873}

sumo_binary = "sumo-gui"

type_params = {"cfm": (num_cars, makecfm())}

env_params = {"target_velocity": 25}

net_params = {"length": 200, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"debug/net/"}

cfg_params = {"type_list": ["rl"], "start_time": 0, "end_time":3000, "cfg_path":"debug/cfg/", "num_cars":num_cars, "type_counts":{"rl": 4}, "use_flows":True, "period":"1"}

scenario = LoopScenario("test-exp", num_cars, type_params, cfg_params, net_params, generator_class=CircleGenerator)

##data path needs to be relative to cfg location

leah_sumo_params = {"port": 8873}

exp = SumoExperiment(SimpleVelocityEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

for _ in range(10):
    exp.env.step([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25])
print(exp.env.get_cars("rl_1", num_back=1, num_forward=1))
for _ in range(20):
    exp.env.step(
        [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
         25, 25, 25, 25])
exp.env.reset()
for _ in range(10):
    exp.env.step(
        [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
         15, 15, 15, 15])

exp.env.terminate()
