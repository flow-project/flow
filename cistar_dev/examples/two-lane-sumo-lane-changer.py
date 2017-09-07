''' Used to test whether sumo lane changing is working as intended

Variables:
    sumo_params {dict} -- [Pass time step, safe mode is on or off
                            human_lc strategic -> make all advantageous lane changes]
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

from cistar.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from cistar.core.vehicles import Vehicles
from cistar.controllers.routing_controllers import *
from cistar.controllers.car_following_models import *
from cistar.core.experiment import SumoExperiment
from cistar.scenarios.loop.gen import CircleGenerator
from cistar.envs.loop import LoopEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(time_step=0.1, human_speed_mode="no_collide", human_lane_change_mode="strategic",
                         sumo_binary="sumo-gui")

vehicles = Vehicles()
vehicles.add_vehicles("idm", (IDMController, {}), None, (ContinuousRouter, {}), 0, 20)

env_params = EnvParams()

additional_net_params = {"length": 200, "lanes": 2, "speed_limit": 35, "resolution": 40}
net_params = NetParams(additional_params=additional_net_params)

initial_config = InitialConfig()

scenario = LoopScenario("single-lane-one-contr", CircleGenerator, vehicles, net_params,
                        initial_config)

env = LoopEnvironment(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(2, 1000)

exp.env.terminate()
