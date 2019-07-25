"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import collections
from datetime import datetime
import os
import sys

import gym
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_pkl


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def bottleneck_visuallizer(args):
    """Visualizer for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_pkl(result_dir)

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if args.run and config_run:
        if args.run != config_run:
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if args.run:
        agent_cls = get_agent_class(args.run)
    elif config_run:
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    sim_params = flow_params['sim']
    sim_params.restart_instance = True
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_path = '{0}/test_time_rollout/'.format(dir_path)
    sim_params.emission_path = emission_path if args.gen_emission else None

    # pick your rendering mode
    if args.render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif args.render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif args.render_mode == 'sumo_gui':
        sim_params.render = True
        print('NOTE: With render mode {}, an extra instance of the SUMO GUI '
              'will display before the GUI for visualizing the result. Click '
              'the green Play arrow to continue.'.format(args.render_mode))
    elif args.render_mode == 'no_render':
        sim_params.render = False
    if args.save_render:
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
        sim_params.save_render = True

    # Start the environment with the gui turned on and a path for the
    # emission file
    env_params = flow_params['env']
    sim_params.restart_instance = False
    if args.evaluate:
        env_params.evaluate = True

    # lower the horizon if testing
    if args.horizon:
        config['horizon'] = args.horizon
        env_params.horizon = args.horizon


    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    # Simulate and collect metrics
    final_outflows = []
    final_inflows = []
    mean_speed = []
    std_speed = []

    policy_agent_mapping = default_policy_agent_mapping
    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    steps = 0
    inflow_grid = list(range(args.outflow_min, args.outflow_max,
                             args.step_size))
    outflow_arr = np.zeros((len(inflow_grid) * args.num_trials, 2))
    # keep track of the last 500 points of velocity data for lane 0
    # and 1 in edge 4
    velocity_arr = np.zeros((args.end_len * len(inflow_grid) * args.num_trials, 3))
    for i in range(len(inflow_grid)):
        for j in range(args.num_trials):
            vel = []
            obs = env.unwrapped.reset(inflow_grid[i])
            mapping_cache = {}  # in case policy_agent_mapping is stochastic
            reward_dict = {}
            agent_states = DefaultMapping(
                lambda agent_id: state_init[mapping_cache[agent_id]])
            prev_actions = DefaultMapping(
                lambda agent_id: action_init[mapping_cache[agent_id]])
            prev_rewards = collections.defaultdict(lambda: 0.)
            done = False
            reward_total = 0.0
            for k in range(env_params.horizon):
                vehicles = env.unwrapped.k.vehicle
                vel.append(np.mean(vehicles.get_speed(vehicles.get_ids())))
                if k >= env_params.horizon - args.end_len:
                    vehs_on_four = vehicles.get_ids_by_edge('4')
                    lanes = vehicles.get_lane(vehs_on_four)
                    lane_dict = {veh_id: lane for veh_id, lane in
                                 zip(vehs_on_four, lanes)}
                    sort_by_lane = sorted(vehs_on_four,
                                          key=lambda x: lane_dict[x])
                    num_zeros = lanes.count(0)
                    if num_zeros > 0:
                        speed_on_zero = np.mean(vehicles.get_speed(
                            sort_by_lane[0:num_zeros]))
                    else:
                        speed_on_zero = 0.0
                    if num_zeros < len(vehs_on_four):
                        speed_on_one = np.mean(vehicles.get_speed(
                            sort_by_lane[num_zeros:]))
                    else:
                        speed_on_one = 0.0
                    velocity_arr[args.end_len * (j + i * args.num_trials) + k - (
                            env_params.horizon - args.end_len), :] = \
                        [inflow_grid[i],
                         speed_on_zero,
                         speed_on_one]
                multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
                action_dict = {}
                for agent_id, a_obs in multi_obs.items():
                    if a_obs is not None:
                        policy_id = mapping_cache.setdefault(
                            agent_id, policy_agent_mapping(agent_id))
                        p_use_lstm = use_lstm[policy_id]
                        if p_use_lstm:
                            a_action, p_state, _ = agent.compute_action(
                                a_obs,
                                state=agent_states[agent_id],
                                prev_action=prev_actions[agent_id],
                                prev_reward=prev_rewards[agent_id],
                                policy_id=policy_id)
                            agent_states[agent_id] = p_state
                        else:
                            a_action = agent.compute_action(
                                a_obs,
                                prev_action=prev_actions[agent_id],
                                prev_reward=prev_rewards[agent_id],
                                policy_id=policy_id)
                        a_action = _flatten_action(a_action)  # tuple actions
                        action_dict[agent_id] = a_action
                        prev_actions[agent_id] = a_action
                action = action_dict

                action = action if multiagent else action[_DUMMY_AGENT_ID]
                next_obs, reward, done, _ = env.step(action)

                if multiagent:
                    for agent_id, r in reward.items():
                        prev_rewards[agent_id] = r
                else:
                    prev_rewards[_DUMMY_AGENT_ID] = reward

                if multiagent:
                    done = done["__all__"]
                    reward_total += sum(reward.values())
                else:
                    reward_total += reward
                steps += 1
                obs = next_obs

        outflow = vehicles.get_outflow_rate(500)
        final_outflows.append(outflow)
        inflow = vehicles.get_inflow_rate(500)
        final_inflows.append(inflow)
        outflow_arr[j + i * args.num_trials, :] = [inflow_grid[i], outflow]
        if np.all(np.array(final_inflows) > 1e-5):
            throughput_efficiency = [x / y for x, y in
                                     zip(final_outflows, final_inflows)]
        else:
            throughput_efficiency = [0] * len(final_inflows)
        mean_speed.append(np.mean(vel))
        std_speed.append(np.std(vel))
        print("Episode reward", reward_total)

    print('==== Summary of results ====')

    print("\nSpeed, mean (m/s): {}".format(mean_speed))
    print('Average, std: {}, {}'.format(np.mean(mean_speed), np.std(
        mean_speed)))
    print("\nSpeed, std (m/s): {}".format(std_speed))
    print('Average, std: {}, {}'.format(np.mean(std_speed), np.std(
        std_speed)))

    # Compute arrival rate of vehicles in the last 500 sec of the run
    print("\nOutflows (veh/hr): {}".format(final_outflows))
    print('Average, std: {}, {}'.format(np.mean(final_outflows),
                                        np.std(final_outflows)))

    # Compute departure rate of vehicles in the last 500 sec of the run
    print("Inflows (veh/hr): {}".format(final_inflows))
    print('Average, std: {}, {}'.format(np.mean(final_inflows),
                                        np.std(final_inflows)))

    # Compute throughput efficiency in the last 500 sec of the
    print("Throughput efficiency (veh/hr): {}".format(throughput_efficiency))
    print('Average, std: {}, {}'.format(np.mean(throughput_efficiency),
                                        np.std(throughput_efficiency)))

    # save the file
    output_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), './trb_data'))
    filename = args.filename
    outflow_name = 'bottleneck_outflow_{}.txt'.format(filename)
    speed_name = 'speed_outflow_{}.txt'.format(filename)
    with open(os.path.join(output_path, outflow_name), 'ab') as file:
        np.savetxt(file, outflow_arr, delimiter=', ')
    with open(os.path.join(output_path, speed_name), 'ab') as file:
        np.savetxt(file, velocity_arr, delimiter=', ')

    # Plot the inflow results
    # open the file and pull from there
    unique_inflows = sorted(list(set(outflow_arr[:, 0])))
    inflows = outflow_arr[:, 0]
    outflows = outflow_arr[:, 1]
    sorted_outflows = {inflow: [] for inflow in unique_inflows}

    for inflow, outflow in zip(inflows, outflows):
        sorted_outflows[inflow].append(outflow)
    mean_outflows = np.asarray([np.mean(sorted_outflows[inflow])
                                for inflow in unique_inflows])
    std_outflows = np.asarray([np.std(sorted_outflows[inflow])
                               for inflow in unique_inflows])

    plt.figure(figsize=(27, 9))
    plt.plot(unique_inflows, mean_outflows, linewidth=2, color='orange')
    plt.fill_between(unique_inflows, mean_outflows - std_outflows,
                     mean_outflows + std_outflows, alpha=0.25, color='orange')
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Outflow' + r'$ \ \frac{vehs}{hour}$')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    plt.minorticks_on()

    # plot the velocity results
    unique_inflows = sorted(list(set(velocity_arr[:, 0])))
    inflows = velocity_arr[:, 0]
    lane_0 = velocity_arr[:, 1]
    lane_1 = velocity_arr[:, 2]
    sorted_vels = {inflow: [] for inflow in unique_inflows}

    for inflow, vel_0, vel_1 in zip(inflows, lane_0, lane_1):
        sorted_vels[inflow] += [vel_0, vel_1]
    mean_vels = np.asarray([np.mean(sorted_vels[inflow])
                            for inflow in unique_inflows])
    std_vels = np.asarray([np.std(sorted_vels[inflow])
                           for inflow in unique_inflows])

    plt.figure(figsize=(27, 9))

    plt.plot(unique_inflows, mean_vels, linewidth=2, color='orange')
    plt.fill_between(unique_inflows, mean_vels - std_vels,
                     mean_vels + std_vels, alpha=0.25, color='orange')
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Velocity' + r'$ \ \frac{m}{s}$')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    plt.minorticks_on()
    plt.show()

    # terminate the environment
    env.unwrapped.terminate()

    # if prompted, convert the emission file into a csv file
    if args.gen_emission:
        time.sleep(0.1)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(env.scenario.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        emission_to_csv(emission_path)

    # if we wanted to save the render, here we create the movie
    if args.save_render:
        dirs = os.listdir(os.path.expanduser('~') + '/flow_rendering')
        # Ignore hidden files
        dirs = [d for d in dirs if d[0] != '.']
        dirs.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d-%H%M%S"))
        recent_dir = dirs[-1]
        # create the movie
        movie_dir = os.path.expanduser('~') + '/flow_rendering/' + recent_dir
        save_dir = os.path.expanduser('~') + '/flow_movies'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        os_cmd = "cd " + movie_dir + " && ffmpeg -i frame_%06d.png"
        os_cmd += " -pix_fmt yuv420p " + dirs[-1] + ".mp4"
        os_cmd += "&& cp " + dirs[-1] + ".mp4 " + save_dir + "/"
        os.system(os_cmd)


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Evaluates a bottleneck agent over a grid of inflows')

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')
    parser.add_argument('filename', type=str, help='Specifies the filename to output the results into.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='sumo_gui',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd, no_render, and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    parser.add_argument('--outflow_min', type=int, default=400, help='Lowest inflow to evaluate over')
    parser.add_argument('--outflow_max', type=int, default=2500, help='Lowest inflow to evaluate over')
    parser.add_argument('--step_size', type=int, default=100, help='The size of increments to sweep over inflow in')
    parser.add_argument('--num_trials', type=int, default=20, help='How many samples of each inflow to take')
    parser.add_argument('--end_len', type=int, default=500, help='How many last seconds of the run to use for '
                                                                 'calculating the outflow statistics')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)
    bottleneck_visuallizer(args)
