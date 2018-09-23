"""Hierarchical loop merge example."""

import os

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import get_registry, register_env as register_rllib_env

from .cooperative_merge import flow_params, HORIZON, make_create_env

# Inner ring distances closest to the merge are range 300-365 (normalized)
fn_choose_subpolicy = """
def choose_policy(inputs):
    return tf.cast(inputs[:, 7] > 0.482496, tf.int32)
"""

if __name__ == "__main__":
    config = ppo.DEFAULT_CONFIG.copy()
    horizon = HORIZON
    num_cpus = 2
    n_rollouts = 3

    ray.init(num_cpus=num_cpus, redirect_output=True)

    config["num_workers"] = num_cpus
    config["timesteps_per_batch"] = horizon * n_rollouts
    config["gamma"] = 0.999  # discount rate

    config["lambda"] = 0.97
    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.00002
    config["num_sgd_iter"] = 30
    config["horizon"] = horizon

    # Two-level policy parameters
    config["model"].update({"fcnet_hiddens": [32, 32]})

    options = {
        "num_subpolicies": 2,
        "fn_choose_subpolicy": fn_choose_subpolicy,
        "hierarchical_fcnet_hiddens": [[32, 32]] * 2
    }
    config["model"].update({"custom_options": options})

    flow_env_name = "TwoLoopsMergePOEnv"
    exp_tag = "merge_two_level_policy_example"
    this_file = os.path.basename(__file__)[:-3]  # filename without '.py'
    flow_params["flowenv"] = flow_env_name
    flow_params["exp_tag"] = exp_tag
    flow_params["module"] = os.path.basename(__file__)[:-3]
    config['model']['custom_options'].update({
        'flowenv': flow_env_name,
        'exp_tag': exp_tag,
        'module': this_file
    })
    create_env, env_name = make_create_env(
        flow_env_name, flow_params, version=0, exp_tag=exp_tag)

    # Register as rllib env
    register_rllib_env(env_name, create_env)

    alg = ppo.PPOAgent(env=env_name, registry=get_registry(), config=config)
    for i in range(2):
        alg.train()
        if i % 20 == 0:
            alg.save()  # save checkpoint
