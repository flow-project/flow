"""Runs the environments located in flow/benchmarks.

The environment file can be modified in the imports to change the environment
this runner script is executed on. This file runs the PPO algorithm in rllib
and utilizes the hyper-parameters specified in:

Proximal Policy Optimization Algorithms by Schulman et. al.
"""
import json

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

# use this to specify the environment to run
from flow.benchmarks.grid1 import flow_params

# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 2

if __name__ == "__main__":
    # get the env name and a creator for the environment
    create_env, env_name = make_create_env(params=flow_params, version=0)

    # initialize a ray instance
    ray.init(redirect_output=True)

    horizon = flow_params["env"].horizon
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = N_ROLLOUTS
    config["timesteps_per_batch"] = horizon * N_ROLLOUTS
    config["vf_loss_coeff"] = 1.0
    config["kl_target"] = 0.02
    config["use_gae"] = True
    config["horizon"] = horizon
    config["clip_param"] = 0.2

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": "PPO",
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 5,
            "max_failures": 999,
            "stop": {
                "training_iteration": 5
            },
            "num_samples": 3,
            # "upload_dir": "s3://bucket"
        },
    })
