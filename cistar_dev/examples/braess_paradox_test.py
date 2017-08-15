"""
Base Braess's Paradox experiment without any automated vehicles.

Human-driven vehicles choose the route in the braess's network that they perceive as having the
shortest travel time. They then update their perception of the travel time of that route using
their most recent observations.
"""

import logging
from cistar.core.exp import SumoExperiment
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *
from cistar.envs.braess_paradox import BraessParadoxEnvironment
from cistar.scenarios.braess_paradox.braess_paradox_scenario import BraessParadoxScenario


logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step": 0.1, "emission_path": "./data/", "human_sm": "no_collide", "human_lc": "no_lat_collide"}

sumo_binary = "sumo-gui"

type_params = {"idm": (30, (IDMController, {}), (StaticLaneChanger, {}), 0)}

env_params = {"close_CD": True}

net_params = {"edge_length": 130, "angle": np.pi/9, "resolution": 40, "lanes": 1,
              "AC_DB_speed_limit": 100, "AD_CB_speed_limit": 10, "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/cfg/"}

scenario = BraessParadoxScenario("braess-paradox", type_params, net_params, cfg_params)

exp = SumoExperiment(BraessParadoxEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 30000)

exp.env.terminate()
