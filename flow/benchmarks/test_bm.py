"""Runs a submitted solution on a specified benchmark.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::
        python test_bm.py bottleneck0 solution_dir/bottleneck_env.py 

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import numpy as np
import os
import sys

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.core.util import get_rllib_config
from flow.core.util import emission_to_csv
from flow.core.experiment import SumoExperiment
import flow.envs
from flow.envs.base_env import Env

from types import MethodType

EXAMPLE_USAGE = """
example usage:
    python test_bm.py solution_dir

Here the argument is:
solution_dir - a directory containig an env and (optionally) a TensorFlow checkpoint
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a Flow Garden solution on a benchmark.",
    epilog=EXAMPLE_USAGE)

# required input parameters
parser.add_argument(
    "solution_dir", type=str, help="File path to solution environment.")

# optional input parameters
parser.add_argument(
    '--num_rollouts',
    type=int,
    default=10,
    help="The number of rollouts to average over.")

if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()
    solution_dir = args.solution_dir

    args_file = open("%s/args.txt"%solution_dir).read()
    sol_args = args_file.split("\n")

    benchmark_name = sol_args[0].strip()
    env_file_name = sol_args[1].strip()
    env_name = sol_args[2].strip()

    rllib_sol = len(sol_args) > 3 and sol_args[3].lower() == 'rllib'
    if rllib_sol:
        agent_cls_name = sol_args[4].strip()
        checkpoint_name = sol_args[5].strip()

    # Import the benchmark and fetch its flow_params
    try:
        benchmark = __import__("flow.benchmarks.%s"%benchmark_name, fromlist=["flow_params"])
    except:
        raise ImportError("Benchmark '%s' does not exist."%benchmark_name)

    flow_params = benchmark.flow_params

    # Recreate the scenario from the named benchmark
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

    # Start the environment 
    env_params = flow_params['env']
    sumo_params = flow_params['sumo']

    # Find the submitted solution environment and instantiate it
    sys.path.append(solution_dir)
    module = __import__(solution_dir, fromlist=[env_file_name])
    env_class = getattr(getattr(module, env_file_name), env_name)
    env = env_class(env_params=env_params, sumo_params=sumo_params, scenario=scenario)

    # Determine a compute_action method. If using RLlib, restore an agent 
    # accordingly and initialize Ray.    
    compute_action = None
    if rllib_sol:
        # Create and register a gym+rllib env
        create_env, gym_env_name = make_create_env(
            params=flow_params, version=0, render=False)
        register_env(gym_env_name, create_env)

        ray.init(num_cpus=3)
        agent_cls = get_agent_class(agent_cls_name)
        agent = agent_cls(env=gym_env_name, config=None)
        checkpoint = solution_dir + '/' + checkpoint_name
        agent._restore(checkpoint)
        compute_action = agent.compute_action
    else:
        compute_action =  env.restore()

    # Ensure the step method and compute_reward method are not redefined
    env.step = MethodType(Env.step, env)
    reward_env = getattr(flow.envs, flow_params['env_name'])
    env.compute_reward = MethodType(reward_env.compute_reward, env)

    # Run the environment in the presence of the pre-trained RL agent for the
    # requested number of time steps / rollouts
    rets = []
    for _ in range(args.num_rollouts):
        state = env.reset()
        done = False
        ret = 0
        for _ in range(env_params.horizon):
            action = compute_action(state)
            state, reward, done, _ = env.step(action)
            ret += reward
            if done:
                break
        rets.append(round(ret, 2))
        print("Return:", round(ret, 2))
    print("Average, std return: {}, {}".format(np.mean(rets), np.std(rets)))

    # terminate the environment
    env.terminate()

    result_file = open("%s/results.txt"%solution_dir, "w")
    result_file.write(str(np.mean(rets).round(4)) + '\n')
    result_file.write(','.join([str(s) for s in rets]))
