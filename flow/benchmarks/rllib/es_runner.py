"""
Runner script for environments located in flow/benchmarks.

The environment file can be modified in the imports to change the environment
this runner script is executed on. Furthermore, the rllib specific algorithm/
parameters can be specified here once and used on multiple environments.
"""
import json

import ray
import ray.rllib.agents.es as es
from ray.tune import run_experiments
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env
from flow.utils.rllib import FlowParamsEncoder

# use this to specify the environment to run
from flow.benchmarks.bottleneck0 import flow_params

# number of rollouts per training iteration
N_ROLLOUTS = 15
# number of parallel workers
N_CPUS = 60

if __name__ == "__main__":
    # get the env name and a creator for the environment
    create_env, env_name = make_create_env(params=flow_params, version=0)

    # initialize a ray instance
    ray.init(redirect_output=True)

    config = es.DEFAULT_CONFIG.copy()
    config["episodes_per_batch"] = N_ROLLOUTS
    config["num_workers"] = N_ROLLOUTS
    config["eval_prob"] = 0.05
    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                           indent=4)
    config['env_config']['flow_params'] = flow_json

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": "ES",
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 25,
            "max_failures": 999,
            "stop": {"training_iteration": 500},
            "num_samples": 3,
            "upload_dir": "s3://public.flow.results/corl_exps/exp_tests3"
        },
    })
