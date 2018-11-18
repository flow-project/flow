"""
Runner script for environments located in flow/benchmarks.
The environment file can be modified in the imports to change the environment
this runner script is executed on. Furthermore, the rllib specific algorithm/
parameters can be specified here once and used on multiple environments.
"""
import json

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune import run_experiments
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env
from flow.utils.rllib import FlowParamsEncoder
from ray.tune import grid_search

# use this to specify the environment to run
from flow.benchmarks.grid1 import flow_params

# number of rollouts per training iteration
N_ROLLOUTS = 50
# number of parallel workers
N_CPUS = 60

if __name__ == "__main__":
    # get the env name and a creator for the environment
    create_env, env_name = make_create_env(params=flow_params, version=0)

    # initialize a ray instance
    ray.init(redirect_output=True)

    alg_run = "ES"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["episodes_per_batch"] = N_ROLLOUTS
    config["num_workers"] = N_ROLLOUTS
    config["eval_prob"] = 0.05
    config["noise_stdev"] = grid_search([0.01, 0.02])
    config["stepsize"] = grid_search([0.01, 0.02])
    config["model"]["fcnet_hiddens"] = [100, 50, 25]
    config["observation_filter"] = "NoFilter"
    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                           indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": alg_run,
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 25,
            "max_failures": 999,
            "stop": {"training_iteration": 500},
            "num_samples": 1,
            "upload_dir": "s3://<BUCKET NAME>"
        },
    })
