"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""

import argparse
from argparse import Namespace
from datetime import datetime
import errno
import json
import os
import pytz
import sys
import subprocess
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

import multilane_ring_smoothing as multilane_ring_config
from examples.exp_configs.rl.multiagent.multiagent_i210 import flow_params as I210_MA_DEFAULT_FLOW_PARAMS
from examples.train import setup_exps_rllib

from flow.core.experiment import Experiment
from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params, get_rllib_config, get_rllib_pkl
from flow.visualize.i210_replay import replay


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
    parser.add_argument('--run_transfer', action='store_true', help="run transfer on i210 to ")

    parser.add_argument('--use_lstm', action='store_true', help='If true, use lstm')
    parser.add_argument('--algorithm', type=str, default="PPO", choices=["PPO", "TD3"],
                        help='Algorithm to run.')
    # parser.add_argument('--grid_search', action='store_true', help='If true, grid search hyperparams')
    # parser.add_argument('--seed_search', action='store_true', help='If true, sweep seed instead of hyperparameter')

    parser.add_argument('--num_lanes', type=int, default=1)
    parser.add_argument('--num_total_veh', type=int, default=22)
    parser.add_argument('--num_av', type=int, default=2)
    parser.add_argument('--ring_length', type=int, default=220)

    parser.add_argument('--simulate', action='store_true', help='If true, simulate instead of train')
    parser.add_argument('--no_render', action='store_true', help='If true, dont render simulation')
    parser.add_argument('--gen_emission', action='store_true', help='If true, generate emission files')

    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Directory with checkpoint to restore training from.')

    return parser.parse_args(args)


if __name__ == "__main__":
    flags = parse_flags(sys.argv[1:])

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")

    flow_params = multilane_ring_config.make_flow_params(1500, flags.num_total_veh,
                                                         flags.num_av, flags.num_lanes, flags.ring_length)

    exp_title = flags.exp_title if flags.exp_title else flow_params["exp_tag"]

    if flags.simulate:
        # Get the custom callables for the runner.
        if hasattr(multilane_ring_config, "custom_callables"):
            callables = multilane_ring_config.custom_callables
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

        policy_graphs = getattr(multilane_ring_config, "POLICY_GRAPHS", None)
        policy_mapping_fn = getattr(multilane_ring_config, "policy_mapping_fn", None)
        policies_to_train = getattr(multilane_ring_config, "policies_to_train", None)

        alg_run, gym_name, config = setup_exps_rllib(
            flow_params, flags.num_cpus, flags.num_rollouts, flags,
            policy_graphs, policy_mapping_fn, policies_to_train)

        # NOTE: Overriding parameters defined in default setup_exps here:

        config["gamma"] = 0.999  # discount rate
        config["model"].update({"fcnet_hiddens": [32, 32, 32]})
        config["use_gae"] = True
        config["lambda"] = 0.97
        config["kl_target"] = 0.02
        config["num_sgd_iter"] = 10
        config["vf_loss_coeff"] = 1e-6
        if flags.use_lstm:
            config['model']['use_lstm'] = True
            config['model']['vf_share_layers'] = True

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

        # Now we add code to loop through the results and create scores of the results
        if flags.run_transfer:
            output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results/'), date), exp_title)
            if not os.path.exists(output_path):
                try:
                    os.makedirs(output_path)
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
            for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser("~/ray_results")):
                if "checkpoint" in dirpath and dirpath.split('/')[-3] == exp_title:

                    folder = os.path.dirname(dirpath)
                    tune_name = folder.split("/")[-1]
                    checkpoint_num = dirpath.split('_')[-1]
                    if int(checkpoint_num) < flags.num_iters:
                        continue
                    rllib_config = get_rllib_pkl(folder)
                    i210_flow_params = deepcopy(I210_MA_DEFAULT_FLOW_PARAMS)

                    args = Namespace(controller=None, run_transfer=None, render_mode=None, gen_emission=None,
                                     evaluate=None, horizon=None, num_rollouts=2, save_render=False, checkpoint_num=checkpoint_num)

                    ray.shutdown()
                    ray.init()

                    replay(args, i210_flow_params, rllib_config=rllib_config, result_dir=folder, output_dir=output_path)

                    if flags.use_s3:
                        for i in range(4):
                            try:
                                p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                                                 "s3://kanaad.experiments/mutli_lane_transfer/{}/{}/{}".format(date,
                                                                                                                                               exp_title,
                                                                                                                                               tune_name)).split(
                                    ' '))
                                p1.wait(50)
                            except Exception as e:
                                print('This is the error ', e)
