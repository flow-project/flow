"""
This script presents the use of two-way intersections in CISTAR.

Cars enter from the bottom and left nodes following a probability distribution, and
continue to move straight until they exit through the top and right nodes, respectively.
"""

from cistar.core.exp import SumoExperiment
from cistar.envs.intersection import SimpleIntersectionEnvironment
from cistar.scenarios.intersections.intersection_scenario import *
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *

logging.basicConfig(level=logging.INFO)

sumo_params = {"port": 8873, "time_step": 0.1, "emission_path": "./data/"}
sumo_binary = "sumo-gui"

type_params = {"idm": (1, (IDMController, {}), (StaticLaneChanger, {}), 0)}

env_params = {"target_velocity": 25, "max-deacc": -6, "max-acc": 3}

net_params = {"horizontal_length_before": 100, "horizontal_length_after": 10, "horizontal_lanes": 1,
              "vertical_length_before": 100, "vertical_length_after": 10, "vertical_lanes": 1,
              "prob_enter": {"bottom": 0.01, "left": 0.01}, "speed_limit": 30, "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "end_time": 3000, "cfg_path": "debug/cfg/"}

scenario = TwoWayIntersectionScenario("figure8", type_params, net_params, cfg_params)

exp = SumoExperiment(SimpleIntersectionEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()
