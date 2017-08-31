"""
Base Braess's Paradox experiment without any automated vehicles.

Human-driven vehicles choose the route in the braess's network that they perceive as having the
shortest travel time. They then update their perception of the travel time of that route using
their most recent observations.
"""

import logging
from cistar.core.experiment import SumoExperiment
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *
from cistar.envs.braess_paradox import BraessParadoxEnvironment
from cistar.scenarios.braess_paradox.braess_paradox_scenario import BraessParadoxScenario
from cistar.core.params import SumoParams
from cistar.core.params import EnvParams

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams()

sumo_binary = "sumo-gui"

type_params = [("idm", 40, (IDMController, {}), (StaticLaneChanger, {}), 0)]

additional_params = {"max-deacc": -6, "max-acc": 3, "close_CD": False}

env_params = EnvParams(additional_params=additional_params)

net_params = {"edge_length": 130, "angle": np.pi/10, "resolution": 40, "lanes": 1,
              "AC_DB_speed_limit": 100, "AD_CB_speed_limit": 10, "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/cfg/"}

scenario = BraessParadoxScenario("braess-paradox", type_params, net_params, cfg_params)

env = BraessParadoxEnvironment(env_params, sumo_binary, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 30000)

exp.env.terminate()
