''' Example of figure 8 with car following models on it. 


Variables:
    sumo_params {dict} -- [Pass time step, where to save data, whether safe mode is on or off]
    sumo_binary {str} -- [Use either sumo-gui or sumo for visual or non-visual]
    type_params {dict} -- [Types of cars in the system. 
    Format {"name": (number, (Model, {params}), (Lane Change Model, {params}), initial_speed)}]
    env_params {dict} -- []
    net_params {dict} -- [Params for network.
                            radius_ring
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
from cistar_dev.scenarios.figure8.figure8_scenario import Figure8Scenario
from cistar_dev.scenarios.loop.loop_scenario import LoopScenario
from cistar_dev.controllers.car_following_models import *
from cistar_dev.controllers.lane_change_controllers import *
from cistar_dev.controllers.rlcontroller import RLController

logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step": 0.1, "emission_path": "./data/", "human_sm": 1}

sumo_binary = "sumo-gui"

type_params = [("idm", 14, (IDMController, {}), (StaticLaneChanger, {}), 0)]

env_params = {"max-deacc": -3, "max-acc": 3}

net_params = {"radius_ring": 30, "lanes": 1, "speed_limit": 30, "resolution": 40,
              "net_path": "debug/net/", "no-internal-links": False}

cfg_params = {"start_time": 0, "end_time": 3000, "cfg_path": "debug/cfg/"}

# initial_config = {"shuffle": False, "bunching": 200}

scenario = Figure8Scenario("figure8", type_params, net_params, cfg_params)

exp = SumoExperiment(SimpleAccelerationEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()
