"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""

import argparse
from datetime import datetime
import json
import os
import pytz
import sys
from time import strftime

import ray
from ray import tune
from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from copy import deepcopy

import multilane_ring_smoothing as exp_config
from examples.train import setup_exps_rllib

from flow.core.experiment import Experiment
from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params


def parse_flags(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    parser.add_argument('--exp_title', type=str, required=False, default=None,
                        help='Informative experiment title to help distinguish results')
    parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--multi_node', action='store_true', help='Set to true if this will '
                                                                  'be run in cluster mode')
    parser.add_argument('--local_mode', action='store_true', help='Set to true if this will '
                                                                  'be run in local mode')
    parser.add_argument('--num_rollouts', type=int, default=1, help='How many rollouts per training batch')
    parser.add_argument('--num_iters', type=int, default=350)
    parser.add_argument('--checkpoint_freq', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=1)
    # parser.add_argument('--grid_search', action='store_true', help='If true, grid search hyperparams')
    # parser.add_argument('--seed_search', action='store_true', help='If true, sweep seed instead of hyperparameter')

    parser.add_argument('--simulate', action='store_true', help='If true, simulate instead of train')
    parser.add_argument('--no_render', action='store_true', help='If true, dont render simulation')
    parser.add_argument('--gen_emission', action='store_true', help='If true, generate emission files')

    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Directory with checkpoint to restore training from.')

    return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    flags = parse_flags(sys.argv[1:])

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")

    flow_params = exp_config.flow_params
    horizon = exp_config.HORIZON

    exp_title = flags.exp_title if flags.exp_title else flow_params["exp_tag"]

    if flags.simulate:
        # Get the custom callables for the runner.
        if hasattr(exp_config, "custom_callables"):
            callables = exp_config.custom_callables
        else:
            callables = None

        # Update some variables based on inputs.
        flow_params['sim'].render = not flags.no_render

        # Specify an emission path if they are meant to be generated.
        if flags.gen_emission:
            flow_params['sim'].emission_path = "./data"

        # Create the experiment object.
        exp = Experiment(flow_params, callables)
        # Run for the specified number of rollouts.
        exp.run(flags.num_rollouts, convert_to_csv=flags.gen_emission)
    else:

        policy_graphs = getattr(exp_config, "POLICY_GRAPHS", None)
        policy_mapping_fn = getattr(exp_config, "policy_mapping_fn", None)
        policies_to_train = getattr(exp_config, "policies_to_train", None)

        alg_run, gym_name, config = setup_exps_rllib(
            flow_params, flags.num_cpus, flags.num_rollouts,
            policy_graphs, policy_mapping_fn, policies_to_train)

        # NOTE: Overriding parameters defined in default setup_exps here:

        config["gamma"] = 0.999  # discount rate
        config["model"].update({"fcnet_hiddens": [32, 32, 32]})
        config["use_gae"] = True
        config["lambda"] = 0.97
        config["kl_target"] = 0.02
        config["num_sgd_iter"] = 10
        config["vf_loss_coeff"] = 1e-6

        if flags.multi_node and flags.local_mode:
            sys.exit("You can't have both local mode and multi node mode on.")

        if flags.multi_node:
            ray.init(redis_address='localhost:6379')
        elif flags.local_mode:
            ray.init(local_mode=True, object_store_memory=200 * 1024 * 1024)
        else:
            ray.init(num_cpus=flags.num_cpus + 1, object_store_memory=200 * 1024 * 1024)

        exp_config = {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": flags.checkpoint_freq,
            "checkpoint_at_end": True,
            "max_failures": 999,
            "stop": {
                "training_iteration": flags.num_iters,
            },
            "num_samples": flags.num_samples
        }

        if flags.use_s3:
            s3_string = 's3://kanaad.experiments/mutli_lane_transfer/' \
                + date + '/' + exp_title
            exp_config['upload_dir'] = s3_string

        if flags.checkpoint_path is not None:
            exp_config['restore'] = flags.checkpoint_path
        trials = run_experiments({exp_title: exp_config})
