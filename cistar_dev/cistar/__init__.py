import logging
from gym.envs.registration import register
from cistar.scenarios.intersections.intersection_scenario import *
from cistar.controllers.rlcontroller import RLController


logger = logging.getLogger(__name__)


register(
    id='SumoEnv-v0',
    entry_point='cistar.core:SumoEnvironment',
)
register(
    id='SimpleIntersectionEnv-v0',
    entry_point='cistar.envs:SimpleIntersectionEnvironment',
)
register(
    id='LoopEnv-v0',
    entry_point='cistar.envs:LoopEnvironment',
)

# Intersection params
sumo_params = {"time_step": 0.1, "emission_path": "./data/", 
                    "starting_position_shuffle": 1, "rl_sm": "aggressive"}
sumo_binary = "sumo"

num_cars = 15

# type_params = {"idm": (1, (IDMController, {}), (StaticLaneChanger, {}), 0)}
type_params = {"rl": (num_cars, (RLController, {}), None, 0.0)}

# 1/intensity is the average time-spacing of the cars
intensity = .3
v_enter = 20.0

env_params = {"target_velocity": v_enter, "max-deacc": -6, "max-acc": 6, 
            "control-length": 150, "max_speed": v_enter}

net_params = {"horizontal_length_in": 600, "horizontal_length_out": 1000, "horizontal_lanes": 1,
              "vertical_length_in": 600, "vertical_length_out": 1000, "vertical_lanes": 1,
              "speed_limit": {"horizontal": 30, "vertical": 30},
              "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "end_time": 3000000, "cfg_path": "debug/cfg/"}

initial_config = {"spacing": "edge_start", "intensity": intensity, "enter_speed": v_enter}

scenario = TwoWayIntersectionScenario("figure8", type_params, net_params, cfg_params, initial_config=initial_config)

register(
    id='TwoIntersectionEnv-v0',
    entry_point='cistar.envs:TwoIntersectionEnvironment',
    kwargs={"env_params": env_params, "sumo_binary": sumo_binary, 
    "sumo_params": sumo_params, "scenario": scenario}
)