"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import numpy as np
import os

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.core.util import get_rllib_config
from flow.core.util import emission_to_csv

EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO

Here the arguments are:
1 - the number of the checkpoint
PPO - the name of the algorithm the code was run with
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a reinforcement learning agent "
    "given a checkpoint.",
    epilog=EXAMPLE_USAGE)

# required input parameters
parser.add_argument(
    "result_dir", type=str, help="Directory containing results")
parser.add_argument("checkpoint_num", type=str, help="Checkpoint number.")

# optional input parameters
parser.add_argument(
    "--run",
    type=str,
    help="The algorithm or model to train. This may refer to "
    "the name of a built-on algorithm (e.g. RLLib's DQN "
    "or PPO), or a user-defined trainable function or "
    "class registered in the tune registry.")
parser.add_argument(
    '--num_rollouts',
    type=int,
    default=1,
    help="The number of rollouts to visualize.")
parser.add_argument(
    '--emission_to_csv',
    action='store_true',
    help='Specifies whether to convert the emission file '
    'created by sumo into a csv file')
parser.add_argument(
    '--evaluate',
    action='store_true',
    help='Specifies whether to use the "evaluate" reward for the environment.')


if __name__ == "__main__":
    args = parser.parse_args()

    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)

    # Run on only one cpu for rendering purposes
    ray.init(num_cpus=1)
    config["num_workers"] = 1

    flow_params = get_flow_params(config)

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(
        params=flow_params, version=0, render=False)
    register_env(env_name, create_env)

    agent_cls = get_agent_class(args.run)
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint-' + args.checkpoint_num
    agent._restore(checkpoint)

    # Recreate the scenario from the pickled parameters
    exp_tag = flow_params["exp_tag"]
    net_params = flow_params['net']
    vehicles = flow_params['veh']
    initial_config = flow_params['initial']
    module = __import__("flow.scenarios", fromlist=[flow_params["scenario"]])
    scenario_class = getattr(module, flow_params["scenario"])
    module = __import__("flow.scenarios", fromlist=[flow_params["generator"]])
    generator_class = getattr(module, flow_params["generator"])

    scenario = scenario_class(
        name=exp_tag,
        generator_class=generator_class,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    # Start the environment with the gui turned on and a path for the
    # emission file
    module = __import__("flow.envs", fromlist=[flow_params["env_name"]])
    env_class = getattr(module, flow_params["env_name"])
    env_params = flow_params['env']
    if args.evaluate:
        env_params.evaluate = True
    sumo_params = flow_params['sumo']
    sumo_params.render = True
    sumo_params.emission_path = "./test_time_rollout/"

    env = env_class(
        env_params=env_params, sumo_params=sumo_params, scenario=scenario)

    # Run the environment in the presence of the pre-trained RL agent for the
    # requested number of time steps / rollouts
    rets = []
    final_outflows = []
    mean_speed = []
    for i in range(args.num_rollouts):
        vel = []
        state = env.reset()
        done = False
        ret = 0
        for _ in range(env_params.horizon):
            vehicles = env.vehicles
            vel.append(np.mean(vehicles.get_speed(vehicles.get_ids())))
            action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            ret += reward
            if done:
                break
        rets.append(ret)
        outflow = vehicles.get_outflow_rate(500)
        final_outflows.append(outflow)
        mean_speed.append(np.mean(vel))
        print("Round {}, Return: {}".format(i, ret))
    print("Average, std return: {}, {}".format(np.mean(rets), np.std(rets)))
    print("Average, std speed: {}, {}".format(np.mean(mean_speed),
                                              np.std(mean_speed)))
    print("Average, std outflow: {}, {}".format(np.mean(final_outflows),
                                                np.std(final_outflows)))

    # terminate the environment
    env.terminate()

    # if prompted, convert the emission file into a csv file
    if args.emission_to_csv:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = "{0}-emission.xml".format(scenario.name)

        emission_path = \
            "{0}/test_time_rollout/{1}".format(dir_path, emission_filename)

        emission_to_csv(emission_path)
