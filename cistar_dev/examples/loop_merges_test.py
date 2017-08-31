"""
(description)
"""

import logging

from cistar.core.params import SumoParams, EnvParams, InitialConfig
from cistar.core.vehicles import Vehicles
from cistar.core.experiment import SumoExperiment

from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *

from cistar.envs.loop_merges import SimpleLoopMergesEnvironment
from cistar.scenarios.loop_merges.gen import LoopMergesGenerator
from cistar.scenarios.loop_merges.loop_merges_scenario import LoopMergesScenario

from numpy import pi

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(time_step=0.1, emission_path="./data/", human_speed_mode="no_collide")

sumo_binary = "sumo-gui"

vehicles = Vehicles()
vehicles.add_vehicles("idm", (IDMController, {}), (StaticLaneChanger, {}), None, 0, 14)
vehicles.add_vehicles("merge-idm", (IDMController, {}), (StaticLaneChanger, {}), None, 0, 14)

additional_env_params = {"target_velocity": 8, "max-deacc": -6, "max-acc": 3, "fail-safe": "None"}
env_params = EnvParams(additional_params=additional_env_params)

net_params = {"merge_in_length": 500, "merge_in_angle": pi/9,
              "merge_out_length": 500, "merge_out_angle": pi * 17/9,
              "ring_radius": 400 / (2 * pi),
              "resolution": 40, "lanes": 1, "speed_limit": 30,
              "net_path": "debug/net/", "no-internal-links": False}

cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/cfg/"}

initial_config = InitialConfig(additional_params={"merge_bunching": 250})

scenario = LoopMergesScenario("loop-merges", LoopMergesGenerator, vehicles, net_params,
                              cfg_params, initial_config=initial_config)

env = SimpleLoopMergesEnvironment(env_params, sumo_binary, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()
