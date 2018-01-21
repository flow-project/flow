''' Used to test out a mixed environment with an IDM controller and
another type of car, in this case our drunk driver class. One lane. 

Variables:
    sumo_params {dict} -- [Pass time step, safe mode is on or off]
    sumo_binary {str} -- [Use either sumo-gui or sumo for visual or non-visual]
    type_params {dict} -- [Types of cars in the system. 
    Format {"name": (number, (Model, {params}), (Lane Change Model, {params}), initial_speed)}]
    env_params {dict} -- [Params for reward function]
    net_params {dict} -- [Params for network.
                            length: road length
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
from flow.envs.trained_policy_env import TrainedPolicyEnv
from flow.core.experiment import SumoExperiment
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.controllers.car_following_models import *
from flow.controllers.trained_policy_controller import TrainedPolicyController
from flow.controllers.lane_change_controllers import *
from flow.controllers.routing_controllers import *
from flow.core.vehicles import Vehicles
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, SumoCarFollowingParams, SumoLaneChangeParams

import os
logging.basicConfig(level=logging.INFO)

# TODO: this hack is gross, but necessary because RlController was renamed to RLCarFollowingController
# if we re-run the aggressive driver experiment this should no longer be necessary
import sys
from flow.controllers import rlcarfollowingcontroller
rlcarfollowingcontroller.RLController = rlcarfollowingcontroller.RLCarFollowingController
sys.modules['flow.controllers.rlcontroller'] = rlcarfollowingcontroller

sumo_params = SumoParams(time_step= 0.1, sumo_binary="sumo-gui")

human_cfm_params = SumoCarFollowingParams(sigma=1.0, tau=3.0)
human_lc_params = SumoLaneChangeParams(
    lcKeepRight=0, lcAssertive=0.5, lcSpeedGain=1.5,
    lcSpeedGainRight=1.0, model="SL2015")

aggressive_cfm_params = SumoCarFollowingParams(
    speedFactor=1.75, decel=7.5, accel=4.5, tau=0.2)
aggressive_lc_params = SumoLaneChangeParams(
    lcAssertive=20, lcPushy=0.8, lcSpeedGain=100.0,
    lcAccelLat=6, lcSpeedGainRight=1.0, model="SL2015")

vehicles = Vehicles()
vehicles.add_vehicles(veh_id="human",
                      acceleration_controller=(SumoCarFollowingController, {}),
                      lane_change_controller=(SumoLaneChangeController, {}),
                      routing_controller=(ContinuousRouter, {}),
                      initial_speed=0,
                      num_vehicles=20,
                      lane_change_mode="custom",
                      custom_lane_change_mode=0b1001010101,
                      sumo_car_following_params=human_cfm_params,
                      sumo_lc_params=human_lc_params)

pkl_file_path = os.path.dirname(os.path.realpath(__file__)) + "/itr_1810.pkl"

vehicles.add_vehicles(
    veh_id="trained",
    acceleration_controller=(TrainedPolicyController, {"pkl_file": pkl_file_path}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1,
    sumo_lc_params=human_lc_params,
    sumo_car_following_params=aggressive_cfm_params,
    lane_change_controller=(SumoLaneChangeController, {}))


env_params = EnvParams(additional_params={"target_velocity":15})
# env_params.fail_safe = "safe_velocity"

additional_net_params = {"length": 500, "lanes": 4, "speed_limit": 15, "resolution": 40}
net_params = NetParams(additional_params=additional_net_params)

initial_config = InitialConfig(spacing="uniform_in_lane", lanes_distribution=4, shuffle=True)

scenario = LoopScenario("single-lane-two-contr", CircleGenerator, vehicles, net_params,
                        initial_config)
# data path needs to be relative to cfg location

env = TrainedPolicyEnv(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

avg_reward = exp.run(1, 1000)

exp.env.terminate()

print(avg_reward)
