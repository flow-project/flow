import argparse
from distutils.util import strtobool
import json
import os
import pickle

import tensorflow as tf

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts

from flow.utils.registry import make_create_env


### FIXME retrieve flowparams from save file instead of hardcoding it
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, IDMController, ContinuousRouter
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=21)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1)
flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_the_ring",

    # name of the flow environment the experiment is running on
    env_name="WaveAttenuationPOEnv",

    # name of the scenarsio class the experiment is running on
    scenario="LoopScenario",

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=5000,
        warmup_steps=750,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [220, 270],
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "length": 260,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40,
        }, ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)



"""
checkpoint_path: path to the checkpoint_<id> folder
                e.g. ~/ray_results/gym/HalfCheetah/v3/2018-12-12T16-48-37-my-sac-experiment-1-0/
                        mujoco-runner_0_seed=7585_2018-12-12_16-48-37xuadh9vd/checkpoint_1000/
render_mode: 'human', 'rgb_array' or None
deterministic: whether or not to evaluate the policy deterministically
"""
def simulate_policy(checkpoint_path, max_path_length=5000, num_rollouts=1,
                    render_mode='human', deterministic=False):
    session = tf.keras.backend.get_session()
    checkpoint_path = checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    # flow_params = variant['flow_params'] TODO
    create_env, _ = make_create_env(params=flow_params, version=0)
    evaluation_environment = create_env()

    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    with policy.set_deterministic(deterministic):
        paths = rollouts(num_rollouts,
                         evaluation_environment,
                         policy,
                         path_length=max_path_length,
                         render_mode=render_mode)

    if render_mode != 'human':
        from pprint import pprint; import pdb; pdb.set_trace()
        pass

    return paths


if __name__ == '__main__':
    simulate_policy("/Users/nathan/ray_results/SAC_ring/2019-05-12T11-06-49-experiment_name/id=28d3ad94-seed=1711_2019-05-12_11-06-49_8i_8w_g/checkpoint_150")
    # TODO cli