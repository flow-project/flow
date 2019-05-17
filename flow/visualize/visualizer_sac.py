"""Visualizer for experiments using the 'softlearning' library (SAC).

Run 'python3.7 visualizer_sac --help' to get example usage
"""

import argparse
import json
import os
import pickle

import tensorflow as tf

from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts

from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.softlearning import adapt_environment_for_sac


EXAMPLE_USAGE = "Example usage:\n\n" + \
    "python3.7 ./visualizer_sac.py /tmp/ray/result_dir 1 " + \
    "--num-rollouts 3 --horizon 3000 --deterministic False\n\n" + \
    "Here the arguments are in order:\n" + \
    "\t\"/tmp/ray/result_dir\" the directory containing the results\n" + \
    "\t\"1\" the number (id) of the checkpoint\n" + \
    "\t\"3\" the number of rollouts to simulate (optional)\n" + \
    "\t\"3000\" the number of steps to roll for each rollout (optional)\n" + \
    "\t\"False\" whether the policy should be made deterministic (optional)\n"


def simulate_policy(args):
    session = tf.keras.backend.get_session()

    experiment_path = args.result_dir
    variant_path = os.path.join(experiment_path, 'params.json')
    checkpoint_dir = f"checkpoint_{args.checkpoint_num}"
    checkpoint_path = os.path.join(experiment_path, checkpoint_dir)

    if not os.path.exists(variant_path):
        raise ValueError(f"Couldn't find file {variant_path}")
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Couldn't find file {checkpoint_path}")

    with open(variant_path, 'r') as f:
        variant = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    flow_params = get_flow_params(variant)
    flow_params['sim'].render = False

    create_env, _ = make_create_env(params=flow_params, version=0)
    evaluation_environment = create_env()
    adapt_environment_for_sac(evaluation_environment)

    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    if args.horizon:
        horizon = args.horizon
    else:
        horizon = variant['algorithm_params']['kwargs']['epoch_length']

    with policy.set_deterministic(args.deterministic):
        paths = rollouts(args.num_rollouts,
                         evaluation_environment,
                         policy,
                         path_length=horizon,
                         render_mode='human')

    return paths


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing the results.')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--num-rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize (by default 1).')
    parser.add_argument(
        '--horizon',
        type=int,
        help='The horizon for each rollout (by default the same as'
             'during the simulation).')
    parser.add_argument(
        '--deterministic',
        type=bool,
        default=False,
        help='Whether or not the policy should be made deterministic'
             '(False by default).')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    simulate_policy(args)
