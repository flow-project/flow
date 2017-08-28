''' Used as example of sugiyama experiment. 22 IDM cars on a ring
create shockwaves 

Variables:
    sumo_params {dict} -- [Pass time step, safe mode is on or off]
    sumo_binary {str} -- [Use either sumo-gui or sumo for visual or non-visual]
    type_params {dict} -- [Types of cars in the system. 
    Format {"name": (number, (Model, {params}), (Lane Change Model, {params}), initial_speed)}]
    env_params {dict} -- [Params for reward function]
    net_params {dict} -- [Params for network.
                            length: road length
                            lanes
                            speed limit
                            resolution: number of edges comprising ring
                            net_path: where to store net]
    cfg_params {dict} -- [description]
    initial_config {dict} -- [shuffle: randomly reorder cars to start experiment
                                spacing: if gaussian, add noise in start positions
                                bunching: how close to place cars at experiment start]
    scenario {[type]} -- [Which road network to use]
'''
import logging
from cistar_dev.core.exp import SumoExperiment
from cistar_dev.envs.loop_accel import SimpleAccelerationEnvironment
from cistar_dev.scenarios.loop.gen import CircleGenerator
from cistar_dev.scenarios.loop.loop_scenario import LoopScenario
from cistar_dev.controllers.car_following_models import *
from cistar_dev.controllers.lane_change_controllers import *

logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step": 0.1, "human_sm": "aggressive"}

sumo_binary = "sumo-gui"

type_params = [("idm", 22, (IDMController, {}), (StaticLaneChanger, {}), 0)]

env_params = {"max-acc": 3, "max-deacc": -6}

net_params = {"length": 230, "lanes": 1, "speed_limit": 30, "resolution": 40, "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "cfg_path": "debug/cfg/"}

initial_config = {"shuffle": False, "bunching": 20}

scenario = LoopScenario("sugiyama", CircleGenerator, type_params, net_params,
                        cfg_params, initial_config)

env = SimpleAccelerationEnvironment(env_params, sumo_binary, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)
