"""
(description)
"""

import logging
from cistar_dev.core.exp import SumoExperiment
from cistar_dev.controllers.car_following_models import *
from cistar_dev.controllers.lane_change_controllers import *
from cistar_dev.envs.loop_merges import SimpleLoopMergesEnvironment
from cistar_dev.scenarios.loop_merges.loop_merges_scenario import LoopMergesScenario

from numpy import pi

logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step": 0.1, "emission_path": "./data/", "human_sm": "no_collide"}

sumo_binary = "sumo-gui"

type_params = {"idm": (14, (IDMController, {}), (StaticLaneChanger, {}), 0),
               "merge-idm": (14, (IDMController, {}), (StaticLaneChanger, {}), 0)}

env_params = {}

net_params = {"merge_in_length": 500, "merge_in_angle": pi/9, "merge_out_length": 500, "merge_out_angle": pi * 17/9,
              "ring_radius": 400 / (2 * pi), "resolution": 40, "lanes": 1, "speed_limit": 30, "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/cfg/"}

initial_config = {"merge_bunching": 250}

scenario = LoopMergesScenario("loop-merges", type_params, net_params, cfg_params, initial_config=initial_config)

exp = SumoExperiment(SimpleLoopMergesEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()
