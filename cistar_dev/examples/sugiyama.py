''' Used as example of sugiyama experiment. 22 IDM cars on a ring
create shockwaves 

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

from cistar.core.experiment import SumoExperiment
from cistar.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from cistar.core.vehicles import Vehicles

from cistar.controllers.routing_controllers import ContinuousRouter
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *

from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.loop.gen import CircleGenerator
from cistar.scenarios.loop.loop_scenario import LoopScenario

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(time_step=0.1, human_speed_mode="aggressive", sumo_binary="sumo-gui")

vehicles = Vehicles()
vehicles.add_vehicles("idm", (IDMController, {}), None, (ContinuousRouter, {}), 0, 22)

additional_env_params = {"target_velocity": 8, "max-deacc": 3, "max-acc": 3, "num_steps": 500}
env_params = EnvParams(additional_params=additional_env_params)

additional_net_params = {"length": 230, "lanes": 1, "speed_limit": 30, "resolution": 40}
net_params = NetParams(additional_params=additional_net_params)

initial_config = InitialConfig(bunching=20)

scenario = LoopScenario("sugiyama", CircleGenerator, vehicles, net_params, initial_config)

env = SimpleAccelerationEnvironment(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)
