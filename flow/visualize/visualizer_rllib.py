import argparse

import numpy as np

import gym
import ray
import ray.rllib.ppo as ppo
from ray.rllib.agent import get_agent_class
from ray.tune.registry import get_registry, register_env as register_rllib_env


EXAMPLE_USAGE = """
example usage:
    ./visualizer_rllib.py /tmp/ray/checkpoint_dir/checkpoint-0 --run PPO
    --flowenv blah
"""


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a reinforcement learning agent "
                "given a checkpoint.", epilog=EXAMPLE_USAGE)

parser.add_argument(
    "checkpoint", type=str, help="Checkpoint from which to evaluate.")
required_named = parser.add_argument_group("required named arguments")
required_named.add_argument(
    "--run", type=str, required=True,
    help="The algorithm or model to train. This may refer to the name "
         "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
         "user-defined trainable function or class registered in the "
         "tune registry.")
required_named.add_argument(
    "--flowenv", type=str, help="The flow example to use.")


if __name__ == "__main__":

    args = parser.parse_args()

    if not args.flowenv:
        if not args.config.get("flowenv"):
            parser.error("the following arguments are required: --flowenv")
        args.flowenv = args.config.get("flowenv")

    ray.init(num_cpus=1)

    cls = get_agent_class(args.run)
    if args.flowenv == "TwoLoopsMergeEnv":
        flow_env_name = "TwoLoopsMergeEnv"
        exp_tag = "two_loops_straight_merge_example"  # experiment prefix
        config = ppo.DEFAULT_CONFIG.copy()
        # TODO(cathywu) load params.json instead
        config["horizon"] = 1000
        config["model"].update({"fcnet_hiddens": [32, 32]})
        config["gamma"] = 0.999
        from examples.rllib.two_loops_straight_merge import make_create_env
    elif args.flowenv == "TwoLoopsMergePOEnv":
        flow_env_name = "TwoLoopsMergePOEnv"
        exp_tag = "two_loops_straight_merge_example"  # experiment prefix
        config = ppo.DEFAULT_CONFIG.copy()
        config["horizon"] = 1000
        config["model"].update({"fcnet_hiddens": [16, 16, 16]})
        config["gamma"] = 0.999
        from examples.rllib.cooperative_merge import make_create_env
    else:
        raise(NotImplementedError, "flowenv %s not supported yet" %
              args.flowenv)

    # Overwrite config for rendering purposes
    config["num_workers"] = 1

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(flow_env_name, version=0,
                                           sumo="sumo")
    register_rllib_env(env_name, create_env)

    agent = cls(env=env_name, registry=get_registry(), config=config)
    agent.restore(args.checkpoint)

    # Create and register a new gym environment for rendering rollout
    create_render_env, env_render_name = make_create_env(flow_env_name,
                                                         version=1,
                                                         sumo="sumo-gui")
    create_render_env()
    env = gym.make(env_render_name)
    for i in range(10):
        state = env.reset()
        done = False
        while not done:
            if isinstance(state, list):
                state = np.concatenate(state)
            action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
