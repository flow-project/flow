"""
(description)
"""
import cloudpickle
import os
import tensorflow as tf

import ray
import ray.rllib.ppo as ppo
from ray.tune.registry import get_registry, register_env as register_rllib_env

HORIZON = 1000


def choose_subpolicy(inputs):
    return tf.cast(inputs[:, 7] > 210, tf.int32)


if __name__ == "__main__":
    config = ppo.DEFAULT_CONFIG.copy()
    horizon = HORIZON
    num_cpus = 1
    n_rollouts = 30

    ray.init(num_cpus=num_cpus, redirect_output=True)

    config["num_workers"] = num_cpus
    config["timesteps_per_batch"] = horizon * n_rollouts
    config["gamma"] = 0.999  # discount rate

    config["lambda"] = 0.97
    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = horizon

    # Two-level policy parameters
    config["model"].update(
        {"fcnet_hiddens": [[32, 32]] * 2})
    config["model"]["user_data"] = {}
    # fn = list(cloudpickle.dumps(lambda x: 0))
    fn = list(cloudpickle.dumps(choose_subpolicy))
    config["model"]["user_data"].update({"num_subpolicies": 2,
                                         "fn_choose_subpolicy": fn})

    flow_env_name = "TwoLoopsMergePOEnv"
    exp_tag = "merge_two_level_policy_example"
    this_file = os.path.basename(__file__)[:-3]  # filename without '.py'
    config['user_data'].update({'flowenv': flow_env_name,
                                'exp_tag': exp_tag,
                                'module': this_file})
    from examples.rllib.cooperative_merge import make_create_env
    create_env, env_name = make_create_env(flow_env_name, version=0,
                                           exp_tag=exp_tag)

    # Register as rllib env
    register_rllib_env(env_name, create_env)

    alg = ppo.PPOAgent(env=env_name, registry=get_registry(), config=config)
    for i in range(2):
        alg.train()
        if i % 20 == 0:
            alg.save()  # save checkpoint
