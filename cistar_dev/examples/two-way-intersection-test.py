"""
This script presents the use of two-way intersections in cistar_dev.

Cars enter from the bottom and left nodes following a probability distribution, and
continue to move straight until they exit through the top and right nodes, respectively.
"""

from cistar.core.experiment import SumoExperiment
from cistar.core.vehicles import Vehicles
from cistar.core.params import SumoParams, EnvParams, InitialConfig, NetParams

from cistar.controllers.car_following_models import *
from cistar.controllers.routing_controllers import *

from cistar.envs.two_intersection import TwoIntersectionEnvironment
from cistar.scenarios.intersections.intersection_scenario import *
from cistar.scenarios.intersections.gen import TwoWayIntersectionGenerator

import logging

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(time_step=0.1, emission_path="./data/", sumo_binary="sumo-gui")

vehicles = Vehicles()
vehicles.add_vehicles("idm", (IDMController, {}), None, None, 0, 20)

intensity = .2
v_enter = 10

env_params = EnvParams(additional_params={"target_velocity": v_enter, "max-deacc": -6, "max-acc": 3,
                                          "control-length": 150, "max_speed": v_enter})

additional_net_params = {"horizontal_length_in": 400, "horizontal_length_out": 10, "horizontal_lanes": 1,
                         "vertical_length_in": 400, "vertical_length_out": 10, "vertical_lanes": 1,
                         "speed_limit": {"horizontal": 30, "vertical": 30}}
net_params = NetParams(no_internal_links=False, additional_params=additional_net_params)

cfg_params = {"start_time": 0, "end_time": 3000, "cfg_path": "debug/cfg/"}

initial_config = InitialConfig(spacing="custom", additional_params={"intensity": intensity, "enter_speed": v_enter})

scenario = TwoWayIntersectionScenario("two-way-intersection", TwoWayIntersectionGenerator,
                                      vehicles, net_params, initial_config=initial_config)

env = TwoIntersectionEnvironment(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()

