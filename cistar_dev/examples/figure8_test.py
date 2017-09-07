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
from cistar.core.experiment import SumoExperiment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.figure8.gen import Figure8Generator
from cistar.scenarios.figure8.figure8_scenario import Figure8Scenario
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *

from cistar.core.params import SumoParams
from cistar.core.params import EnvParams

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams()

sumo_binary = "sumo"

type_params = [("idm", 14, (IDMController, {}), (StaticLaneChanger, {}), 0)]

additional_params = {"max-deacc": -3, "max-acc": 3}

env_params = EnvParams(additional_params=additional_params)

net_params = {"radius_ring": 30, "lanes": 1, "speed_limit": 30, "resolution": 40,
              "net_path": "debug/net/", "no-internal-links": False}

cfg_params = {"start_time": 0, "end_time": 3000, "cfg_path": "debug/cfg/"}

# initial_config = {"shuffle": False, "bunching": 200}

scenario = Figure8Scenario("figure8", Figure8Generator, type_params, net_params, cfg_params)

env = SimpleAccelerationEnvironment(env_params, sumo_binary, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()
