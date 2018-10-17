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
import yaml

from types import MethodType
import importlib.util as imp

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
import flow.envs
from flow.envs.base_env import Env

EXAMPLE_USAGE = """
example usage:
    python test_bm.py solution_dir

Here the argument is:
solution_dir - a directory containing the submission env and auxiliary files
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
    args = parser.parse_args()
    solution_dir = args.solution_dir

    # Automatically create file indicating failure, to be overwritten
    # at the end of the script if everything runs correctly
    result_file = open(solution_dir + "/results.txt", "w")
    result_file.write('FAILED\n')

    try:
        # Parse arguments
        config_path = solution_dir + "/solution_config."
        try:
            solution_config = yaml.load(open(config_path + "yaml"))
        except Exception as e:
            try:
                solution_config = yaml.load(open(config_path + "yml"))
            except Exception as e:
                raise RuntimeError("Failed to find and load solution_config" +
                                   ".yaml or solution_config.yml")

        benchmark_name = solution_config['benchmark']
        env_file_path = solution_config['env_file_path']
        if env_file_path[-3:] != '.py':
            raise ValueError("Filepath for env does not point to a " +
                             "python file.")
        split = 0
        if '/' in env_file_path:
            split = env_file_path.rfind('/') + 1
        env_file_name = env_file_path[split:-3]
        env_name = solution_config['env_name']

        rllib_sol = 'rllib_agent_type' in solution_config \
            and 'checkpoint_name' in solution_config
        if rllib_sol:
            agent_cls_name = solution_config['rllib_agent_type']
            checkpoint_name = solution_config['checkpoint_name']

        # Import the benchmark and fetch its flow_params
        benchmark = __import__(
            "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
        flow_params = benchmark.flow_params

        # Recreate the scenario from the named benchmark
        exp_tag = flow_params["exp_tag"]
        net_params = flow_params['net']
        vehicles = flow_params['veh']
        initial_config = flow_params['initial']
        module = __import__(
            "flow.scenarios", fromlist=[flow_params["scenario"]])
        scenario_class = getattr(module, flow_params["scenario"])
        module = __import__(
            "flow.scenarios", fromlist=[flow_params["generator"]])
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
        full_env_path = solution_dir + '/' + env_file_path
        spec = imp.spec_from_file_location(env_file_name, full_env_path)
        module = imp.module_from_spec(spec)
        spec.loader.exec_module(module)
        env_class = getattr(module, env_name)

        env = env_class(
            env_params=env_params, sumo_params=sumo_params, scenario=scenario)

        # Determine a compute_action method. If using RLlib, restore an agent
        # accordingly and initialize Ray.
        compute_action = None
        if rllib_sol:
            # Create and register a gym+rllib env using flow params from
            # the named benchmark
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
            compute_action = env.restore()

        # Ensure the step method and compute_reward method are not redefined
        env.step = MethodType(Env.step, env)
        reward_env = getattr(flow.envs, flow_params['env_name'])
        env.compute_reward = MethodType(reward_env.compute_reward, env)

        # Run the environment in the presence of the pre-trained RL agent for
        # the requested number of time steps / rollouts
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
            print("Reward:", round(ret, 2))
        print("Average, std return: {}, {}".format(
            np.mean(rets), np.std(rets)))

        # terminate the environment
        env.terminate()

        result_file = open(solution_dir + "/results.txt", "w")
        result_file.write(str(np.mean(rets).round(4)) + '\n')
        result_file.write(','.join([str(s) for s in rets]))

    except Exception as e:
        result_file.write(str(e))
