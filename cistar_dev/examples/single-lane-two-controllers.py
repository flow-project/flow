''' Used to test out a mixed environment with an IDM controller and
another type of car, in this case our drunk driver class. One lane. 

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
from cistar.envs.loop import LoopEnvironment
from cistar.core.exp import SumoExperiment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *

logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step":0.1, "human_sm": "no_collide"}

sumo_binary = "sumo-gui"

type_params = {"idm": (15, (IDMController, {}), (StaticLaneChanger, {}), 0), 
                "idm2": (1, (DrunkDriver, {}), (StaticLaneChanger, {}), 0)}

env_params = {}

net_params = {"length": 200, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"debug/net/"}

cfg_params = {"start_time": 0, "end_time":30000, "cfg_path":"debug/cfg/"}

initial_config = {"shuffle": False, "bunching": 20å}

scenario = LoopScenario("single-lane-two-contr", type_params, net_params, cfg_params, initial_config)
##data path needs to be relative to cfg location

exp = SumoExperiment(LoopEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

# 400 runs, 1000 steps per run
exp.run(400, 1000)

exp.env.terminate()
