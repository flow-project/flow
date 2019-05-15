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

from flow.utils.rllib import get_flow_params


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

    flow_params = get_flow_params(variant)
    flow_params['sim'].render = True
    print(flow_params)

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