import argparse
import json
import importlib

import numpy as np

import gym
import ray
import ray.rllib.ppo as ppo
from ray.rllib.agent import get_agent_class
from ray.tune.registry import get_registry, register_env as register_rllib_env


EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a reinforcement learning agent "
                "given a checkpoint.", epilog=EXAMPLE_USAGE)

# parser.add_argument(
#     "checkpoint", type=str, help="Checkpoint from which to evaluate.")
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
# required_named.add_argument(
#     "--flowenv", type=str, required=True,
#     help="The flow example to use.")
optional_named = parser.add_argument_group("optional named arguments")
optional_named.add_argument(
    '--num_rollouts', type=int, default=10,
    help="The number of rollouts to visualize.")


if __name__ == "__main__":

    args = parser.parse_args()

    # if not args.flowenv:
    #     if not args.config.get("flowenv"):
    #         parser.error("the following arguments are required: --flowenv")
    #     args.flowenv = args.config.get("flowenv")

    # find json file somehow
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
                    else args.result_dir[:-1]
    jsonfile = result_dir + '/params.json'
    jsondata = json.loads(open(jsonfile).read())
    gamma = jsondata['gamma']
    horizon = jsondata['horizon']
    hidden_layers = jsondata['model']['fcnet_hiddens']
    user_data = jsondata['user_data']
    flow_env_name = user_data['flowenv']
    exp_tag = user_data['exp_tag']
    module_name = 'examples.rllib.' + user_data['module']
    env_module = importlib.import_module(module_name)
    make_create_env = env_module.make_create_env

    ray.init(num_cpus=1)

    config = ppo.DEFAULT_CONFIG.copy()        
    config['horizon'] = horizon
    # config["model"].update({"fcnet_hiddens": hidden_layers})
    config["model"] = jsondata["model"]
    config["gamma"] = gamma

    # Overwrite config for rendering purposes
    config["num_workers"] = 1

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(flow_env_name, version=0,
                                           sumo="sumo")
    register_rllib_env(env_name, create_env)

    agent_cls = get_agent_class(args.run)
    agent = agent_cls(env=env_name, registry=get_registry(), config=config)
    checkpoint = result_dir + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    # Create and register a new gym environment for rendering rollout
    create_render_env, env_render_name = make_create_env(flow_env_name,
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
