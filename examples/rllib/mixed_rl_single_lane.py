"""
Basic implementation of a mixed-rl single-lane environment with accelerations as
actions for the autonomous vehicles.
"""
import logging

import gym
import numpy as np

import ray
import ray.rllib.ppo as ppo
from ray.tune.registry import get_registry, register_env as register_rllib_env
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models import ModelCatalog

from flow.core.util import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.controllers.rlcarfollowingcontroller import RLCarFollowingController
from flow.controllers.lane_change_controllers import StaticLaneChanger
from flow.controllers.car_following_models import CFMController
from flow.controllers.routing_controllers import ContinuousRouter


ray.init()
# ray.init(redirect_output=True)
config = ppo.DEFAULT_CONFIG.copy()
config["num_sgd_itr"] = 20

flow_env_name = "AccelEnv"

env_version_num = 0
env_name = flow_env_name + '-v' + str(env_version_num)


def create_env(env_config):
    import flow.envs as flow_envs
    logging.basicConfig(level=logging.INFO)

    tot_cars = 8
    auton_cars = 4
    human_cars = tot_cars - auton_cars

    sumo_params = SumoParams(sim_step=0.1, sumo_binary="sumo")

    vehicles = Vehicles()
    vehicles.add("rl", (RLCarFollowingController, {}), (StaticLaneChanger, {}),
                 (ContinuousRouter, {}), 0, auton_cars)
    vehicles.add("cfm", (CFMController, {}), (StaticLaneChanger, {}),
                 (ContinuousRouter, {}), 0, human_cars)

    additional_env_params = {"target_velocity": 8, "num_steps": 1000}
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = {"length": 200, "lanes": 1, "speed_limit": 30, "resolution": 40}
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig()

    scenario = LoopScenario("rl-test", CircleGenerator, vehicles, net_params, initial_config)

    pass_params = (flow_env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    register_env(*pass_params)
    env = gym.envs.make(env_name)

    return env

register_rllib_env(env_name, create_env)

alg = ppo.PPOAgent(env=env_name, registry=get_registry())
for i in range(20):
    alg.train()
