"""Run the environments located in flow/benchmarks using TRPO."""
import os
import argparse
import json
import sys
from time import strftime

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO

from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder
from flow.core.util import ensure_dir


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python trpo_runner.py BENCHMARK_NAME")

    # required input parameters
    parser.add_argument(
        'benchmark_name', type=str,
        help='Name of the experiment configuration file, as located in '
             'flow/benchmarks.')

    # optional input parameters
    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use.')
    parser.add_argument(
        '--num_steps', type=int, default=9000000,
        help='How many total steps to perform learning over.')
    parser.add_argument(
        '--rollout_size', type=int, default=30000,
        help='How many steps are in a training batch.')

    return parser.parse_known_args(args)[0]


def run_model(params, rollout_size=50, num_steps=50):
    """Perform the training operation.

    Parameters
    ----------
    params : dict
        flow-specific parameters (see flow/utils/registry.py)
    rollout_size : int
        length of a single rollout
    num_steps : int
        total number of training steps

    Returns
    -------
    stable_baselines.*
        the trained model
    """
    constructor = env_constructor(params, version=0)()
    env = DummyVecEnv([lambda: constructor])

    model = TRPO(
        'MlpPolicy',
        env,
        verbose=2,
        timesteps_per_batch=rollout_size,
        gamma=0.999,
        policy_kwargs={
            "net_arch": [100, 50, 25]
        },
    )
    model.learn(total_timesteps=num_steps)

    return model


def save_model(model, params, save_path):
    """Save the trained model and flow-specific parameters.

    Parameters
    ----------
    model : stable_baselines.*
        the trained model
    params : dict
        flow-specific parameters (see flow/utils/registry.py)
    save_path : str
        path to saved model and experiment configuration files
    """
    print('Saving the trained model!')

    # save the trained model
    model.save(os.path.join(save_path, "model"))

    # dump the flow params
    with open(os.path.join(save_path, 'flow_params.json'), 'w') as outfile:
        json.dump(
            params, outfile, cls=FlowParamsEncoder, sort_keys=True, indent=4)


if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])

    # Import the benchmark and fetch its flow_params
    module = __import__("flow.benchmarks.{}".format(flags.benchmark_name),
                        fromlist=["flow_params"])
    flow_params = module.flow_params

    # Path to the saved files
    exp_tag = flow_params['exp_tag']
    result_name = '{}/{}'.format(exp_tag, strftime("%Y-%m-%d-%H:%M:%S"))

    # Define the save path and ensure that the required directories exist.
    dir_path = os.path.realpath(os.path.expanduser('~/baseline_results'))
    ensure_dir(dir_path)
    path = os.path.join(dir_path, result_name)

    # Perform the training operation.
    train_model = run_model(flow_params, flags.rollout_size, flags.num_steps)

    # Save the model to a desired folder and then delete it to demonstrate
    # loading.
    save_model(train_model, flow_params, path)
