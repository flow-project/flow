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
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import ray

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import gym

import flow.envs
from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl

EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /tmp/ray/result_dir 1

Here the arguments are:
1 - the number of the checkpoint
"""

OUTFLOW_RANGE = [400, 2500]
STEP_SIZE = 100
NUM_TRIALS = 20
END_LEN = 500


class _RLlibPreprocessorWrapper(gym.ObservationWrapper):
    """Adapts a RLlib preprocessor for use as an observation wrapper."""

    def __init__(self, env, preprocessor):
        super(_RLlibPreprocessorWrapper, self).__init__(env)
        self.preprocessor = preprocessor

        from gym.spaces.box import Box
        self.observation_space = Box(
            -1.0, 1.0, preprocessor.shape, dtype=np.float32)

    def observation(self, observation):
        return self.preprocessor.transform(observation)


def bottleneck_visualizer(args):
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    # config = get_rllib_config(result_dir + '/..')
    # pkl = get_rllib_pkl(result_dir + '/..')
    config = get_rllib_config(result_dir)
    # TODO(ev) backwards compatibility hack
    try:
        pkl = get_rllib_pkl(result_dir)
    except Exception:
        pass

    # check if we have a multiagent scenario but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policy_graphs', {}):
        multiagent = True
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)

    # hack for old pkl files
    # TODO(ev) remove eventually
    sim_params = flow_params['sim']
    setattr(sim_params, 'num_clients', 1)

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(
        params=flow_params, version=0, render=False)
    register_env(env_name, create_env)

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

    sim_params.restart_instance = True

    sim_params.emission_path = './test_time_rollout/'

    # prepare for rendering
    if args.render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif args.render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif args.render_mode == 'sumo_gui':
        sim_params.render = True
    elif args.render_mode == 'no_render':
        sim_params.render = False
        sim_params.restart_instance = True

    if args.save_render:
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
        sim_params.save_render = True

    # Recreate the scenario from the pickled parameters
    exp_tag = flow_params['exp_tag']
    net_params = flow_params['net']
    vehicles = flow_params['veh']
    initial_config = flow_params['initial']
    module = __import__('flow.scenarios', fromlist=[flow_params['scenario']])
    scenario_class = getattr(module, flow_params['scenario'])

    scenario = scenario_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    # check if the environment is a single or multiagent environment, and
    # get the right address accordingly
    single_agent_envs = [env for env in dir(flow.envs)
                         if not env.startswith('__')]

    if flow_params['env_name'] in single_agent_envs:
        env_loc = 'flow.envs'
    else:
        env_loc = 'flow.multiagent_envs'

    # Start the environment with the gui turned on and a path for the
    # emission file
    module = __import__(env_loc, fromlist=[flow_params['env_name']])
    env_class = getattr(module, flow_params['env_name'])
    env_params = flow_params['env']
    if args.evaluate:
        env_params.evaluate = True

    # lower the horizon if testing
    if args.horizon:
        config['horizon'] = args.horizon
        env_params.horizon = args.horizon

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    _env = env_class(
        env_params=env_params, sim_params=sim_params, scenario=scenario)
    _prep = ModelCatalog.get_preprocessor(_env, options={})
    env = _RLlibPreprocessorWrapper(_env, _prep)

    if config['model']['use_lstm']:
        use_lstm = True
        state_init = [
            np.zeros(config['model']['lstm_cell_size'], np.float32),
            np.zeros(config['model']['lstm_cell_size'], np.float32)
        ]
    else:
        use_lstm = False

    if multiagent:
        rets = {}
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn'].func
        for key in config['multiagent']['policy_graphs'].keys():
            rets[key] = []
    else:
        rets = []
    final_outflows = []
    mean_speed = []

    inflow_grid = list(range(OUTFLOW_RANGE[0], OUTFLOW_RANGE[1],
                             STEP_SIZE))
    outflow_arr = np.zeros((len(inflow_grid) * NUM_TRIALS, 2))
    # keep track of the last 500 points of velocity data for lane 0
    # and 1 in edge 4
    velocity_arr = np.zeros((END_LEN * len(inflow_grid) * NUM_TRIALS, 3))

    for i in range(len(inflow_grid)):
        for j in range(NUM_TRIALS):
            vel = []
            hidden_state_keys = {}
            state = env.unwrapped.reset(inflow_grid[i])
            if multiagent:
                ret = {key: [0] for key in rets.keys()}
            else:
                ret = 0
            for k in range(env_params.horizon):
                vehicles = env.unwrapped.k.vehicle
                vel.append(np.mean(vehicles.get_speed(vehicles.get_ids())))
                if k >= env_params.horizon - END_LEN:
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
                    velocity_arr[END_LEN * (j + i * NUM_TRIALS) + k - (
                            env_params.horizon - END_LEN), :] = \
                        [inflow_grid[i],
                            speed_on_zero,
                            speed_on_one]

                if multiagent:
                    action = {}
                    for agent_id in state.keys():
                        if use_lstm:
                            if agent_id not in hidden_state_keys.keys():
                                hidden_state_keys[agent_id] = [
                                    np.zeros(config['model']['lstm_cell_size'], np.float32),
                                    np.zeros(config['model']['lstm_cell_size'], np.float32)
                                ]
                            action[agent_id], hidden_state_keys[agent_id],\
                            logits = agent.compute_action(
                                state[agent_id], state=hidden_state_keys[agent_id],
                                policy_id=policy_map_fn(agent_id))
                        else:
                            action[agent_id] = agent.compute_action(
                                state[agent_id], policy_id=policy_map_fn(agent_id))
                else:
                    action = agent.compute_action(state)
                state, reward, done, _ = env.step(action)
                if multiagent:
                    for actor, rew in reward.items():
                        ret[policy_map_fn(actor)][0] += rew
                else:
                    ret += reward
                if multiagent and done['__all__']:
                    break
                if not multiagent and done:
                    break

            if multiagent:
                for key in rets.keys():
                    rets[key].append(ret[key])
            else:
                rets.append(ret)
            outflow = vehicles.get_outflow_rate(500)
            outflow_arr[j + i * NUM_TRIALS, :] = [inflow_grid[i], outflow]
            final_outflows.append(outflow)
            mean_speed.append(np.mean(vel))
            if multiagent:
                for agent_id, rew in rets.items():
                    print('Round {}, Return: {} for agent {}'.format(
                        j, ret, agent_id))
            else:
                print('Round {}, Return: {}'.format(j, ret))
        if multiagent:
            for agent_id, rew in rets.items():
                print('Average, std return: {}, {} for agent {}'.format(
                    np.mean(rew), np.std(rew), agent_id))
        else:
            print('Average, std return: {}, {}'.format(
                np.mean(rets), np.std(rets)))
    print('Average, std speed: {}, {}'.format(
        np.mean(mean_speed), np.std(mean_speed)))
    print('Average, std outflow: {}, {}'.format(
        np.mean(final_outflows), np.std(final_outflows)))

    # terminate the environment
    env.unwrapped.terminate()

    # save the file
    output_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), './data'))
    if args.filename:
        filename = args.filename
        outflow_name = '/bottleneck_outflow_{}.txt'.format(filename)
        speed_name = '/speed_outflow_{}.txt'.format(filename)
        np.savetxt(output_path + outflow_name,
                   outflow_arr, delimiter=', ')
        np.savetxt(output_path + speed_name,
                   velocity_arr, delimiter=', ')
    else:
        np.savetxt(output_path + '/bottleneck_outflow_MA_LC_LSTM.txt',
                   outflow_arr, delimiter=', ')
        np.savetxt(output_path + '/speed_outflow_MA_LC_LSTM.txt',
                   velocity_arr, delimiter=', ')

    # Plot the inflow results
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

    # if prompted, convert the emission file into a csv file
    if args.emission_to_csv:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(scenario.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        emission_to_csv(emission_path)

    # if we wanted to save the render, here we create the movie
    if args.save_render:
        dirs = os.listdir(os.path.expanduser('~') + '/flow_rendering')
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

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
        '--num-rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--emission-to-csv',
        action='store_true',
        help='Specifies whether to convert the emission file '
             'created by sumo into a csv file')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='no_render',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd, no_render and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='saves the render to a file')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    parser.add_argument(
        '--filename',
        type=str,
        help='Specifies the filename to output the results into.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)
    bottleneck_visualizer(args)
