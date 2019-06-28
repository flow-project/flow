"""Visualizer for rllab-trained experiments."""

import argparse
import joblib
import numpy as np
import os

from flow.core.util import emission_to_csv

from rllab.sampler.utils import rollout


def visualizer_rllab(args):
    """Visualizer for rllab experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    # extract the flow environment
    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']

    # FIXME(ev, ak) only one of these should be needed
    # unwrapped_env = env._wrapped_env._wrapped_env.env.unwrapped
    # unwrapped_env = env.wrapped_env.env.env.unwrapped

    # if this doesn't work, try the one above it
    unwrapped_env = env._wrapped_env.env.unwrapped

    # Set sumo to make a video
    sim_params = unwrapped_env.sim_params
    sim_params.emission_path = './test_time_rollout/' if args.gen_emission \
        else None
    if args.no_render:
        sim_params.render = False
    else:
        sim_params.render = True
    unwrapped_env.restart_simulation(
        sim_params=sim_params, render=sim_params.render)

    # Load data into arrays
    rew = []
    for j in range(args.num_rollouts):
        # run a single rollout of the experiment
        path = rollout(env=env, agent=policy)

        # collect the observations and rewards from the rollout
        new_rewards = path['rewards']

        # print the cumulative reward of the most recent rollout
        print('Round {}, return: {}'.format(j, sum(new_rewards)))
        rew.append(sum(new_rewards))

    # print the average cumulative reward across rollouts
    print('Average, std return: {}, {}'.format(np.mean(rew), np.std(rew)))

    # if prompted, convert the emission file into a csv file
    if args.gen_emission:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(
            unwrapped_env.scenario.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        emission_to_csv(emission_path)


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=100,
        help='Number of rollouts we will average over')
    parser.add_argument(
        '--no_render',
        action='store_true',
        help='Whether to render the result')
    parser.add_argument(
        '--plotname',
        type=str,
        default='traffic_plot',
        help='Prefix for all generated plots')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    visualizer_rllab(args)
