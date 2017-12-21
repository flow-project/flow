''' Basic test of fully rl environment with accelerations as actions. Single lane. Mixed human and rl.

Variables:
    sumo_params {dict} -- [Pass time step, whether safe mode is on or off]
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

import gym
from gym.envs.registration import register

import ray
import ray.rllib.ppo as ppo
from ray.tune.registry import get_registry, register_env
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models import ModelCatalog

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.controllers.routing_controllers import *

from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.controllers.rlcontroller import RLController
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *

from flow.core.params import SumoParams
from flow.core.params import EnvParams

ray.init(redirect_output=True)
config = ppo.DEFAULT_CONFIG.copy()
config["num_sgd_itr"] = 20

flow_env_name = "SimpleAccelerationEnvironment"

env_version_num = 0
env_name = flow_env_name+'-v'+str(env_version_num)


class TuplePreprocessor(Preprocessor):

    def _init(self):
        self.shape = self._obs_space.shape

    def transform(self, observation):
        return np.concatenate(observation)


def pass_params_to_gym(env_name, sumo_params, type_params, env_params, net_params,
                initial_config, scenario):
    global env_version_num

    num_steps = 500
    if "num_steps" in env_params.additional_params:
        num_steps = env_params.additional_params["num_steps"]
    register(
        id=env_name+'-v'+str(env_version_num),
        entry_point='flow.envs:'+env_name,
        max_episode_steps=num_steps,
        kwargs={"env_params": env_params, "sumo_params": sumo_params, "scenario": scenario}
    )
    #env_version_num += 1



def create_env():
    import flow.envs as flow_envs
    logging.basicConfig(level=logging.INFO)

    tot_cars = 8
    auton_cars = 4
    human_cars = tot_cars - auton_cars

    sumo_params = SumoParams(time_step=0.1, human_speed_mode="no_collide", rl_speed_mode="no_collide",
                             sumo_binary="sumo")

    vehicles = Vehicles()
    vehicles.add_vehicles("rl", (RLController, {}), (StaticLaneChanger, {}), (ContinuousRouter, {}), 0, auton_cars)
    vehicles.add_vehicles("cfm", (CFMController, {}), (StaticLaneChanger, {}), (ContinuousRouter, {}), 0, human_cars)

    additional_env_params = {"target_velocity": 8, "num_steps": 1000}
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = {"length": 200, "lanes": 1, "speed_limit": 30, "resolution": 40}
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig()

    scenario = LoopScenario("rl-test", CircleGenerator, vehicles, net_params, initial_config)

    pass_params = (flow_env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    # env = GymEnv(env_name, record_video=False, register_params=pass_params)

    pass_params_to_gym(*pass_params)
    env = gym.envs.make(env_name)

    env.observation_space.shape = (
    int(np.sum([c.shape for c in env.observation_space.spaces])),)

    # env.spec.id = env_name
    print("XXXXXXX Created", env)
    ModelCatalog.register_preprocessor(env_name, TuplePreprocessor)

    return env

register_env(env_name, create_env)
# ModelCatalog.register_preprocessor(env_name, TuplePreprocessor)

alg = ppo.PPOAgent(env=env_name, registry=get_registry())
for i in range(20):
    alg.train()

# alg = ppo.PPOAgent(config=config, env="CartPole-v1")
