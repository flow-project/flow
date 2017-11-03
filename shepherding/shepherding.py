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
from sheppherding_env import SheppherdingEnv
from flow.core.experiment import SumoExperiment
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.controllers.routing_controllers import *
from flow.core.vehicles import Vehicles
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, SumoCarFollowingParams, SumoLaneChangeParams

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(time_step= 0.1, sumo_binary="sumo-gui")

human_cfm_params = SumoCarFollowingParams(sigma=1.0, speedDev=0.1)
human_lc_params = SumoLaneChangeParams(lcKeepRight=0, model="SL2015")
aggressive_cfm_params = SumoCarFollowingParams(speedFactor=2.0, maxSpeed=30, minGap=0.05, decel=7.5, tau=0.1)
aggressive_lc_params = SumoLaneChangeParams(lcAssertive=20, lcPushy=0.8, lcSpeedGain=2.0, model="SL2015")

vehicles = Vehicles()
vehicles.add_vehicles("human", (SumoCarFollowingController, {}), (SumoLaneChangeController, {}), (ContinuousRouter, {}),
                      0, 40, lane_change_mode="custom", custom_lane_change_mode=0b1000010101,
                      sumo_car_following_params=human_cfm_params,
                      sumo_lc_params=human_lc_params)
vehicles.add_vehicles("aggressive-human", (SumoCarFollowingController, {}), (SumoLaneChangeController, {}),
                      (ContinuousRouter, {}), 0, 1, lane_change_mode="execute_all",
                      sumo_car_following_params=aggressive_cfm_params,
                      sumo_lc_params=aggressive_lc_params)

env_params = EnvParams(additional_params={"target_velocity":30})
# env_params.fail_safe = "safe_velocity"

additional_net_params = {"length": 300, "lanes": 3, "speed_limit": 20, "resolution": 40}
net_params = NetParams(additional_params=additional_net_params)

initial_config = InitialConfig(spacing="uniform_random", scale=3, lanes_distribution=3, shuffle=True)

scenario = LoopScenario("single-lane-two-contr", CircleGenerator, vehicles, net_params,
                        initial_config)
# data path needs to be relative to cfg location

env = SheppherdingEnv(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 5000)

exp.env.terminate()
