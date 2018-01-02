import argparse

import gym
import ray
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
    flow_env_name = "TwoLoopsMergeEnv"
    exp_tag = "two_loops_straight_merge_example"  # experiment prefix
    from examples.rllib.two_loops_straight_merge import make_create_env
    create_env, env_name = make_create_env(flow_env_name, version=0,
                                           exp_tag=exp_tag, sumo="sumo")
    # Register as rllib env
    register_rllib_env(env_name, create_env)

    agent = cls(env=env_name, registry=get_registry())
    agent.restore(args.checkpoint)

    env = gym.make(env_name)
    state = env.reset()
    done = False
    while args.loop_forever or not done:
        action = agent.compute_action(state)
        state, reward, done, _ = env.step(action)
