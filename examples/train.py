"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""

import argparse
import json
import os
import sys
from time import strftime

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

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

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params


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
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')

    # optional input parameters
    parser.add_argument(
        '--rl_trainer', type=str, default="RLlib",
        help='the RL trainer to use. either RLlib or Stable-Baselines')

    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=5000,
        help='How many total steps to perform learning over')
    parser.add_argument(
        '--rollout_size', type=int, default=1000,
        help='How many steps are in a training batch.')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory with checkpoint to restore training from.')

    return parser.parse_known_args(args)[0]


def run_model_stablebaseline(flow_params, num_cpus=1, rollout_size=50, num_steps=50):
    """Run the model for num_steps if provided.

    Parameters
    ----------
    num_cpus : int
        number of CPUs used during training
    rollout_size : int
        length of a single rollout
    num_steps : int
        total number of training steps
    The total rollout length is rollout_size.

    Returns
    -------
    stable_baselines.*
        the trained model
    """
    if num_cpus == 1:
        constructor = env_constructor(params=flow_params, version=0)()
        # The algorithms require a vectorized environment to run
        env = DummyVecEnv([lambda: constructor])
    else:
        env = SubprocVecEnv([env_constructor(params=flow_params, version=i)
                             for i in range(num_cpus)])

    train_model = PPO2('MlpPolicy', env, verbose=1, n_steps=rollout_size)
    train_model.learn(total_timesteps=num_steps)
    return train_model


def setup_exps_rllib(flow_params,
                     n_cpus,
                     n_rollouts,
                     policy_graphs=None,
                     policy_mapping_fn=None,
                     policies_to_train=None):
    """Return the relevant components of an RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters (see flow/utils/registry.py)
    n_cpus : int
        number of CPUs to run the experiment over
    n_rollouts : int
        number of rollouts per training iteration
    policy_graphs : dict, optional
        TODO
    policy_mapping_fn : function, optional
        TODO
    policies_to_train : list of str, optional
        TODO

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    horizon = flow_params['env'].horizon

    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    config["num_workers"] = n_cpus
    config["train_batch_size"] = horizon * n_rollouts
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32, 32]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["horizon"] = horizon

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # multiagent configuration
    if policy_graphs is not None:
        print("policy_graphs", policy_graphs)
        config['multiagent'].update({'policies': policy_graphs})
    if policy_mapping_fn is not None:
        config['multiagent'].update({'policy_mapping_fn': tune.function(policy_mapping_fn)})
    if policies_to_train is not None:
        config['multiagent'].update({'policies_to_train': policies_to_train})

    create_env, gym_name = make_create_env(params=flow_params)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])

    # import relevant information from the exp_config script
    module = __import__("exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    module_ma = __import__("exp_configs.rl.multiagent", fromlist=[flags.exp_config])
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
        assert flags.rl_trainer == "RLlib", \
            "Currently, multiagent experiments are only supported through "\
            "RLlib. Try running this experiment using RLlib: 'python train.py EXP_CONFIG'"
    else:
        assert False, "Unable to find experiment config!"
    if flags.rl_trainer == "RLlib":
        flow_params = submodule.flow_params
        n_cpus = submodule.N_CPUS
        n_rollouts = submodule.N_ROLLOUTS
        policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
        policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
        policies_to_train = getattr(submodule, "policies_to_train", None)

        alg_run, gym_name, config = setup_exps_rllib(
            flow_params, n_cpus, n_rollouts,
            policy_graphs, policy_mapping_fn, policies_to_train)

        ray.init(num_cpus=n_cpus + 1, object_store_memory=200 * 1024 * 1024)
        exp_config = {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "checkpoint_at_end": True,
            "max_failures": 999,
            "stop": {
                "training_iteration": flags.num_steps,
            },
        }

        if flags.checkpoint_path is not None:
            exp_config['restore'] = flags.checkpoint_path
        trials = run_experiments({flow_params["exp_tag"]: exp_config})

    elif flags.rl_trainer == "Stable-Baselines":
        flow_params = submodule.flow_params
        # Path to the saved files
        exp_tag = flow_params['exp_tag']
        result_name = '{}/{}'.format(exp_tag, strftime("%Y-%m-%d-%H:%M:%S"))

        # Perform training.
        print('Beginning training.')
        model = run_model_stablebaseline(flow_params, flags.num_cpus, flags.rollout_size, flags.num_steps)

        # Save the model to a desired folder and then delete it to demonstrate
        # loading.
        print('Saving the trained model!')
        path = os.path.realpath(os.path.expanduser('~/baseline_results'))
        ensure_dir(path)
        save_path = os.path.join(path, result_name)
        model.save(save_path)

        # dump the flow params
        with open(os.path.join(path, result_name) + '.json', 'w') as outfile:
            json.dump(flow_params, outfile,
                      cls=FlowParamsEncoder, sort_keys=True, indent=4)

        # Replay the result by loading the model
        print('Loading the trained model and testing it out!')
        model = PPO2.load(save_path)
        flow_params = get_flow_params(os.path.join(path, result_name) + '.json')
        flow_params['sim'].render = True
        env_constructor = env_constructor(params=flow_params, version=0)()
        # The algorithms require a vectorized environment to run
        eval_env = DummyVecEnv([lambda: env_constructor])
        obs = eval_env.reset()
        reward = 0
        for _ in range(flow_params['env'].horizon):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = eval_env.step(action)
            reward += rewards
        print('the final reward is {}'.format(reward))
    else:
        assert False, "rl_trainer should be either 'RLlib' or 'Stable-Baselines'!"
