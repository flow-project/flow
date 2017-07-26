"""
Base Braess's Paradox experiment without any automated vehicles
"""

import logging
from cistar.core.exp import SumoExperiment
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *
from cistar.envs.braess_paradox import BraessParadoxEnvironment
from cistar.scenarios.braess_paradox.braess_paradox_scenario import BraessParadoxScenario


logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step": 0.1, "emission_path": "./data/", "human_sm": "no_collide"}

sumo_binary = "sumo-gui"

type_params = {"idm": (22, (IDMController, {}), (StaticLaneChanger, {}), 0)}

env_params = {"varying_edge_speed": lambda density: 0.51 / density, "constant_edge_speed": 7.2}

net_params = {"edge_length": 120, "angle": np.pi/9, "resolution": 40, "lanes": 1, "speed_limit": 30,
              "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/cfg/"}

scenario = BraessParadoxScenario("braess-paradox", type_params, net_params, cfg_params)

exp = SumoExperiment(BraessParadoxEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 30000)

exp.env.terminate()
