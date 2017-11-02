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
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(time_step= 0.1, sumo_binary="sumo-gui")

vehicles = Vehicles()
vehicles.add_vehicles("human", (IDMController, {"v0":10}), (SumoLaneChangeController, {}), (ContinuousRouter, {}), 0, 53, lane_change_mode="execute_all")
vehicles.add_vehicles("aggressive-human", (IDMController, {"v0":40}), (SumoLaneChangeController, {}), (ContinuousRouter, {}), 0, 1, lane_change_mode="execute_all")

env_params = EnvParams(additional_params={"target_velocity":30})
# env_params.fail_safe = "safe_velocity"

additional_net_params = {"length": 800, "lanes": 5, "speed_limit": 20, "resolution": 40}
net_params = NetParams(additional_params=additional_net_params)

initial_config = InitialConfig(spacing="uniform_random", scale=40, lanes_distribution=5)

scenario = LoopScenario("single-lane-two-contr", CircleGenerator, vehicles, net_params,
                        initial_config)
# data path needs to be relative to cfg location

env = SheppherdingEnv(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 5000)

exp.env.terminate()
