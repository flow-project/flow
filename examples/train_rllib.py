"""Runner script for single and multi-agent RLlib experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train_rllib.py EXP_CONFIG
"""
import sys
import json
import argparse
import ray
from ray import tune
from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.utils.rllib import FlowParamsEncoder
from flow.utils.registry import make_create_env
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class


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
        epilog="python train_rllib.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/singleagent' or 'exp_configs/multiagent.')

    return parser.parse_known_args(args)[0]


def setup_exps(flow_params,
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
    config = agent_cls._default_config.copy()
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
        config.update({'multiagent': {'policy_graphs': policy_graphs}})
    if policy_mapping_fn is not None:
        config.update({'multiagent': {'policy_mapping_fn': tune.function(policy_mapping_fn)}})
    if policies_to_train is not None:
        config.update({'multiagent': {'policies_to_train': policies_to_train}})

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])

    # import relevant information from the exp_config script
    module = __import__("exp_configs.singleagent", fromlist=[flags.exp_config])
    module_ma = __import__("exp_configs.multiagent", fromlist=[flags.exp_config])
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
    else:
        assert False, "Unable to find experiment config!"
    flow_params = submodule.flow_params
    n_cpus = submodule.N_CPUS
    n_rollouts = submodule.N_ROLLOUTS
    policy_graphs = getattr(submodule, "policy_graphs", None)
    policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
    policies_to_train = getattr(submodule, "policies_to_train", None)

    alg_run, gym_name, config = setup_exps(
        flow_params, n_cpus, n_rollouts,
        policy_graphs, policy_mapping_fn, policies_to_train)

    ray.init(num_cpus=n_cpus + 1)
    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "checkpoint_at_end": True,
            "max_failures": 999,
            "stop": {
                "training_iteration": 200,
            },
        }
    })
