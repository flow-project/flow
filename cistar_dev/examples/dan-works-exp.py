''' Used as proof that car trying to move at constant velocity can stabilize the ring 

Variables:
    sumo_params {dict} -- [Pass time step, whether safe mode is on or off]
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
from cistar_dev.envs.loop import LoopEnvironment
from cistar_dev.scenarios.loop.loop_scenario import LoopScenario
from cistar_dev.controllers.car_following_models import *
from cistar_dev.controllers.velocity_controllers import *
from cistar_dev.controllers.lane_change_controllers import *

logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step":0.01, "human_sm": 1}

sumo_binary = "sumo-gui"

type_params = {"constantV": (1, (ConstantVelocityController, {"constant_speed": 3.5}), (StaticLaneChanger, {}), 0),
               "idm": (21, (IDMController, {}), (StaticLaneChanger, {}), 0)}

env_params = {}

net_params = {"length": 230, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"dan-work/net/"}

cfg_params = {"start_time": 0, "end_time":50000, "cfg_path":"dan-work/cfg/"}

initial_config = {"shuffle":False, "bunching": 20}

scenario = LoopScenario("test-exp", type_params, net_params, cfg_params, initial_config)
##data path needs to be relative to cfg location

exp = SumoExperiment(LoopEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 10000)
