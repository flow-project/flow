"""Runs the environments located in flow/benchmarks.

The environment file can be modified in the imports to change the environment
this runner script is executed on. This file runs the ARS algorithm in rllib
and utilizes the hyper-parameters specified in:

Simple random search provides a competitive approach to reinforcement learning
by Mania et. al
"""
import json

import ray
import ray.rllib.ars as ars
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

# use this to specify the environment to run
from flow.benchmarks.figureeight2 import flow_params

# number of rollouts per training iteration
N_ROLLOUTS = 25
# number of parallel workers
PARALLEL_ROLLOUTS = 25

if __name__ == "__main__":
    # get the env name and a creator for the environment
    create_env, env_name = make_create_env(params=flow_params, version=0)

    # initialize a ray instance
    ray.init(redis_address="localhost:6379", redirect_output=True)

    config = ars.DEFAULT_CONFIG.copy()
    config["num_workers"] = PARALLEL_ROLLOUTS
    config["num_deltas"] = N_ROLLOUTS
    config["deltas_used"] = grid_search([25, 50])
    config["sgd_stepsize"] = .01
    config["delta_std"] = grid_search([.01, .02])
    config['policy'] = 'Linear'
    config["observation_filter"] = "NoFilter"
    config['eval_rollouts'] = PARALLEL_ROLLOUTS

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": "ARS",
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 5,
            "max_failures": 999,
            "stop": {
                "training_iteration": 500
            },
            "repeat": 3,
            "trial_resources": {
                "cpu": 1,
                "gpu": 0,
                "extra_cpu": PARALLEL_ROLLOUTS - 1,
            },
        },
    })
