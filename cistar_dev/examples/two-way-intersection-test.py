"""
This script presents the use of two-way intersections in cistar.

Cars enter from the bottom and left nodes following a probability distribution, and
continue to move straight until they exit through the top and right nodes, respectively.
"""

from cistar.core.exp import SumoExperiment
from cistar.envs.two_intersection import TwoIntersectionEnvironment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.intersections.gen import TwoWayIntersectionGenerator
from cistar.scenarios.intersections.intersection_scenario import *
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *

import logging

logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step": 0.1, "emission_path": "./data/"}
sumo_binary = "sumo-gui"

type_params = [("idm", 20, (IDMController, {}), None, 0)]

intensity = .2
v_enter = 10

env_params = {"target_velocity": v_enter, "max-deacc": -6, "max-acc": 3,
              "control-length": 150, "max_speed": v_enter}

net_params = {"horizontal_length_in": 400, "horizontal_length_out": 10, "horizontal_lanes": 1,
              "vertical_length_in": 400, "vertical_length_out": 10, "vertical_lanes": 1,
              "speed_limit": {"horizontal": 30, "vertical": 30},
              "net_path": "debug/net/", "no-internal-links": False}

cfg_params = {"start_time": 0, "end_time": 3000, "cfg_path": "debug/cfg/"}

initial_config = {"spacing": "custom", "intensity": intensity, "enter_speed": v_enter}

scenario = TwoWayIntersectionScenario("two-way-intersection", TwoWayIntersectionGenerator,
                                      type_params, net_params, cfg_params, initial_config=initial_config)

env = TwoIntersectionEnvironment(env_params, sumo_binary, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()

