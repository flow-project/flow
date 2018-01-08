"""
Visualizer for rllib experimenst

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::
    
        python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO

optional_named : ArgumentGroup
    Optional named command-line arguments
parser : ArgumentParser
    Command-line argument parser
required_named : ArgumentGroup
    Required named command-line arguments
"""

import argparse
import json
import importlib

import numpy as np

import gym
import ray
import ray.rllib.ppo as ppo
from ray.rllib.agent import get_agent_class
from ray.tune.registry import get_registry, register_env as register_rllib_env

from flow.core.util import unstring_flow_params, get_rllib_params, get_flow_params


EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO
OR
    python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO \
        --module cooperative_merge --flowenv TwoLoopsMergePOEnv \
        --exp_tag cooperative_merge_example    
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a reinforcement learning agent "
                "given a checkpoint.", epilog=EXAMPLE_USAGE)

parser.add_argument(
    "result_dir", type=str, help="Directory containing results")
parser.add_argument(
    "checkpoint_num", type=str, help="Checkpoint number.")

required_named = parser.add_argument_group("required named arguments")
required_named.add_argument(
    "--run", type=str, required=True,
    help="The algorithm or model to train. This may refer to the name "
         "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
         "user-defined trainable function or class registered in the "
         "tune registry.")

optional_named = parser.add_argument_group("optional named arguments")
optional_named.add_argument(
    '--num_rollouts', type=int, default=10,
    help="The number of rollouts to visualize.")
optional_named.add_argument(
    '--module', type=str, default = '',
    help='Location of the make_create_env function to use')
optional_named.add_argument(
    '--flowenv', type=str, default = '',
    help='Flowenv being used')
optional_named.add_argument(
    '--exp_tag', type=str, default = '',
    help='Experiment tag')

if __name__ == "__main__":

    args = parser.parse_args()

    result_dir = args.result_dir if args.result_dir[-1] != '/' \
                    else args.result_dir[:-1]

    rllib_params = get_rllib_params(result_dir)

    gamma = rllib_params['gamma']
    horizon = rllib_params['horizon']
    hidden_layers = rllib_params['hidden_layers']

    if args.module:
        module_name = 'examples.rllib.' + args.module
        env_module = importlib.import_module(module_name)
        
        make_create_env = env_module.make_create_env
        flow_params = env_module.flow_params

        flow_env_name = args.flowenv
        exp_tag = args.exp_tag
    else:
        flow_params, make_create_env = get_flow_params(result_dir)

        flow_env_name = flow_params['flowenv']
        exp_tag = flow_params['exp_tag']

    ray.init(num_cpus=1)

    config = ppo.DEFAULT_CONFIG.copy()        
    config['horizon'] = horizon
    # config["model"].update({"fcnet_hiddens": hidden_layers})
    config["model"] = jsondata["model"]
    config["gamma"] = gamma

    # Overwrite config for rendering purposes
    config["num_workers"] = 1

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(flow_env_name, flow_params,
                                           version=0, sumo="sumo")
    register_rllib_env(env_name, create_env)

    agent_cls = get_agent_class(args.run)
    agent = agent_cls(env=env_name, registry=get_registry(), config=config)
    checkpoint = result_dir + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    # Create and register a new gym environment for rendering rollout
    create_render_env, env_render_name = make_create_env(flow_env_name,
                                                         flow_params,
                                                         version=1,
                                                         sumo="sumo-gui")
    env = create_render_env()
    env_num_steps = env.env.env_params.additional_params['num_steps']
    if env_num_steps != horizon:
        print("WARNING: mismatch of experiment horizon and rendering horizon "
              "{} != {}".format(horizon, env_num_steps))
    rets = []
    for i in range(args.num_rollouts):
        state = env.reset()
        done = False
        ret = 0
        while not done:
            if isinstance(state, list):
                state = np.concatenate(state)
            action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            ret += reward
        rets.append(ret)
        print("Return:", ret)
    print("Average Return", np.mean(rets))
