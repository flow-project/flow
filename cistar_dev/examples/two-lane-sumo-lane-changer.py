''' Used to test whether sumo lane changing is working as intended

Variables:
    sumo_params {dict} -- [Pass time step, safe mode is on or off
                            human_lc strategic -> make all advantageous lane changes]
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
from cistar_dev.controllers.lane_change_controllers import *

logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step": 0.1, "human_sm": "no_collide", "human_lc": "strategic"}

sumo_binary = "sumo-gui"

type_params = [("idm", 20, (IDMController, {}), None, 0)]

env_params = {}

net_params = {"length": 200, "lanes": 2, "speed_limit": 35, "resolution": 40, "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "end_time": 3000, "cfg_path": "debug/cfg/"}


initial_config = {"shuffle": True}

scenario = LoopScenario("single-lane-one-contr", type_params, net_params, cfg_params, initial_config)
##data path needs to be relative to cfg location

exp = SumoExperiment(LoopEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

exp.run(2, 1000)

exp.env.terminate()
