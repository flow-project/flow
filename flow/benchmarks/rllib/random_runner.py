"""Runs the environments located in flow/benchmarks using a random agent.
"""
import json
import argparse

import numpy as np

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

EXAMPLE_USAGE = """
example usage:
    python sac_runner.py --benchmark_name=grid0
Here the arguments are:
benchmark_name - name of the benchmark to run
num_rollouts - number of rollouts to train across
num_cpus - number of cpus to use for training
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a Flow Garden solution on a benchmark.",
    epilog=EXAMPLE_USAGE)

# required input parameters
parser.add_argument(
    "--upload_dir", type=str, help="S3 Bucket to upload to.")
# required input parameters
parser.add_argument(
    "--grid_search", action='store_true', default=False, help="Whether to grid search hyperparameters")
parser.add_argument(
    "--num_runs", type=int, default=1, help="How may times to run the agents")

# required input parameters
parser.add_argument(
    "--benchmark_name", type=str, help="File path to solution environment.")

if __name__ == "__main__":
    benchmark_name = 'grid0'
    args = parser.parse_args()
    # benchmark name
    benchmark_name = args.benchmark_name

    upload_dir = args.upload_dir

    # Import the benchmark and fetch its flow_params
    benchmark = __import__(
        "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.flow_params

    # get the env name and a creator for the environment
    create_env, env_name = make_create_env(params=flow_params, version=0)

    env = create_env()
    total_rewards =[]
    for i in range(args.num_runs):
        done = False
        step_num = 0
        episode_rew = 0
        low = env.action_space.low
        high = env.action_space.high
        _ = env.reset()
        while step_num < flow_params['env'].horizon and not done:
            obs, rew, done, info = env.step(np.random.uniform(low=low, high=high))
            episode_rew += rew
        total_rewards.append(episode_rew)

    print('Average rew {}'.format(np.mean(total_rewards)))