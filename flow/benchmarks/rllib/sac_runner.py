"""Runs the environments located in flow/benchmarks.

The environment file can be modified in the imports to change the environment
this runner script is executed on. This file runs the PPO algorithm in rllib
and utilizes the hyper-parameters specified in:
Proximal Policy Optimization Algorithms by Schulman et. al.
"""
import json
import argparse

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

    # initialize a ray instance
    ray.init()

    alg_run = "SAC"

    horizon = flow_params["env"].horizon
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    if args.grid_search:
        config['prioritized_replay'] = grid_search([True, False])
        config['target_network_update_freq'] = grid_search([0, 10])
        config['optimization']['actor_learning_rate'] = grid_search([3e-3, 3e-4])
        config['optimization']['critic_learning_rate'] = grid_search([3e-3, 3e-4])

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # Register as rllib env
    register_env(env_name, create_env)

    exp_tag = {
        "run": alg_run,
        "env": env_name,
        "config": {
            **config
        },
        "checkpoint_freq": 100,
        "max_failures": 999,
        "stop": {
            "training_iteration": 500
        },
        "num_samples": 1,

    }

    if upload_dir:
        exp_tag["upload_dir"] = "s3://" + upload_dir

    trials = run_experiments({
        flow_params["exp_tag"]: exp_tag
    })
