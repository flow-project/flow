"""Transfer and replay for i210 environment."""
import argparse
from copy import deepcopy
import numpy as np
import os
import time

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl

from flow.visualize.transfer.util import inflows_range

from examples.exp_configs.rl.multiagent.multiagent_i210 import flow_params as I210_MA_DEFAULT_FLOW_PARAMS

EXAMPLE_USAGE = """
example usage:
    python ./i210_replay.py /ray_results/experiment_dir/result_dir 1
    python ./i210_replay.py --controller idm
    python ./i210_replay.py --controller idm --run_transfer

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


def replay(args, flow_params, transfer_test=None, rllib_config=None, result_dir=None):
    """Replay or run transfer test (defined by transfer_fn) by modif.

    Arguments:
    ---------
        args {[Namespace]} -- [args from argparser]
        flow_params {[flow_params object, pulled from ]} -- [description]
        transfer_fn {[type]} -- [description]

    Keyword Arguments:
    -----------------
        rllib_config {[type]} -- [description] (default: {None})
        result_dir {[type]} -- [description] (default: {None})
    """
    assert bool(args.controller) ^ bool(rllib_config), \
        "Need to specify either controller or rllib_config, but not both"

    if args.run_transfer:
        flow_params = transfer_test.flow_params_modifier_fn(flow_params)

    if args.controller:
        test_params = {}
        if args.controller == 'idm':
            from flow.controllers.car_following_models import IDMController
            controller = IDMController
            test_params.update({'v0': 1, 'T': 1, 'a': 0.2, 'b': 0.2})  # An example of really obvious changes
        elif args.controller == 'sumo':
            from flow.controllers.car_following_models import SimCarFollowingController
            controller = SimCarFollowingController
        
        flow_params['veh'].type_parameters['av']['acceleration_controller'] = (controller, test_params)

        for veh_param in flow_params['veh'].initial:
            if veh_param['veh_id'] == 'av':
                veh_param['acceleration_controller'] = (controller, test_params)

    sim_params = flow_params['sim']
    sim_params.num_clients = 1

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
        sim_params.render = False  # will be set to True below
    elif args.render_mode == 'no_render':
        sim_params.render = False
    if args.save_render:
        if args.render_mode != 'sumo_gui':
            sim_params.render = 'drgb'
            sim_params.pxpm = 4
        sim_params.save_render = True

    # Start the environment with the gui turned on and a path for the
    # emission file
    env_params = flow_params['env']
    env_params.restart_instance = False
    if args.evaluate:
        env_params.evaluate = True

    # lower the horizon if testing
    if args.horizon:
        env_params.horizon = args.horizon

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(params=flow_params, version=0)
    env = create_env(env_name)

    if args.render_mode == 'sumo_gui':
        env.sim_params.render = True  # set to True after initializing agent and env

    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    if rllib_config:
        # check if we have a multiagent environment but in a
        # backwards compatible way
        if rllib_config.get('multiagent', {}).get('policies', None):
            multiagent = True
            pkl = get_rllib_pkl(result_dir)
            rllib_config['multiagent'] = pkl['multiagent']
        else:
            multiagent = False
            raise NotImplementedError

        # Run on only one cpu for rendering purposes
        rllib_config['num_workers'] = 0

        # lower the horizon if testing
        if args.horizon:
            rllib_config['horizon'] = args.horizon

        assert 'run' in rllib_config['env_config'], "Was this trained with the latest version of Flow?"
        # Determine agent and checkpoint
        config_run = rllib_config['env_config']['run']

        rllib_flow_params = get_flow_params(rllib_config)
        agent_create_env, agent_env_name = make_create_env(params=rllib_flow_params, version=0)
        register_env(agent_env_name, agent_create_env)
        agent_cls = get_agent_class(config_run)

        # create the agent that will be used to compute the actions
        agent = agent_cls(env=agent_env_name, config=rllib_config)
        checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
        checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
        agent.restore(checkpoint)

        if multiagent:
            rets = {}
            # map the agent id to its policy
            policy_map_fn = rllib_config['multiagent']['policy_mapping_fn']
            for key in rllib_config['multiagent']['policies'].keys():
                rets[key] = []
        else:
            rets = []

        if rllib_config['model']['use_lstm']:
            use_lstm = True
            if multiagent:
                state_init = {}
                # map the agent id to its policy
                policy_map_fn = rllib_config['multiagent']['policy_mapping_fn']
                size = rllib_config['model']['lstm_cell_size']
                for key in rllib_config['multiagent']['policies'].keys():
                    state_init[key] = [np.zeros(size, np.float32),
                                       np.zeros(size, np.float32)]
            else:
                state_init = [
                    np.zeros(rllib_config['model']['lstm_cell_size'], np.float32),
                    np.zeros(rllib_config['model']['lstm_cell_size'], np.float32)
                ]
        else:
            use_lstm = False

    # Simulate and collect metrics
    final_outflows = []
    final_inflows = []
    mean_speed = []
    std_speed = []
    for i in range(args.num_rollouts):
        vel = []
        state = env.reset()
        # don't think we're interested in per-agent rewards in this case, just system level rewards
        # but i could be wrong. either way, would have to do this a bit differently for per-agent rewards
        # for non-RL vehicles
        # if rllib_config:
        #     if multiagent:
        #         ret = {key: [0] for key in rets.keys()}
        #     else:
        #         ret = 0
        for _ in range(env_params.horizon):
            vehicles = env.unwrapped.k.vehicle
            speeds = vehicles.get_speed(vehicles.get_ids())

            # only include non-empty speeds
            if speeds:
                vel.append(np.mean(speeds))

            if rllib_config:
                if multiagent:
                    action = {}
                    for agent_id in state.keys():
                        if use_lstm:
                            action[agent_id], state_init[agent_id], logits = \
                                agent.compute_action(
                                state[agent_id], state=state_init[agent_id],
                                policy_id=policy_map_fn(agent_id))
                        else:
                            action[agent_id] = agent.compute_action(
                                state[agent_id], policy_id=policy_map_fn(agent_id))
                else:
                    action = agent.compute_action(state)
            else:
                action = None

            state, reward, done, _ = env.step(action)
            # see above comment
            # if multiagent:
            #     for actor, rew in reward.items():
            #         ret[policy_map_fn(actor)][0] += rew
            # else:
            #     ret += reward
            if type(done) is dict and done['__all__']:
                break
            elif type(done) is not dict and done:
                break

        # if multiagent:
        #     for key in rets.keys():
        #         rets[key].append(ret[key])
        # else:
        #     rets.append(ret)
        outflow = vehicles.get_outflow_rate(500)
        final_outflows.append(outflow)
        inflow = vehicles.get_inflow_rate(500)
        final_inflows.append(inflow)
        if np.all(np.array(final_inflows) > 1e-5):
            throughput_efficiency = [x / y for x, y in
                                     zip(final_outflows, final_inflows)]
        else:
            throughput_efficiency = [0] * len(final_inflows)
        mean_speed.append(np.mean(vel))
        std_speed.append(np.std(vel))
        # if multiagent:
        #     for agent_id, rew in rets.items():
        #         print('Round {}, Return: {} for agent {}'.format(
        #             i, ret, agent_id))
        # else:
        #     print('Round {}, Return: {}'.format(i, ret))

    print('======== Summary of results ========')
    if args.run_transfer:
        print("Transfer test: {}".format(transfer_test.transfer_str))
    print("====================================")
    print("Return:")
    print(mean_speed)
    # if multiagent:
    #     for agent_id, rew in rets.items():
    #         print('For agent', agent_id)
    #         print(rew)
    #         print('Average, std return: {}, {} for agent {}'.format(
    #             np.mean(rew), np.std(rew), agent_id))
    # else:
    #     print(rets)
    #     print('Average, std: {}, {}'.format(
    #         np.mean(rets), np.std(rets)))

    print("\nSpeed, mean (m/s):")
    print(mean_speed)
    print('Average, std: {}, {}'.format(np.mean(mean_speed), np.std(
        mean_speed)))
    print("\nSpeed, std (m/s):")
    print(std_speed)
    print('Average, std: {}, {}'.format(np.mean(std_speed), np.std(
        std_speed)))

    # Compute arrival rate of vehicles in the last 500 sec of the run
    print("\nOutflows (veh/hr):")
    print(final_outflows)
    print('Average, std: {}, {}'.format(np.mean(final_outflows),
                                        np.std(final_outflows)))
    # Compute departure rate of vehicles in the last 500 sec of the run
    print("Inflows (veh/hr):")
    print(final_inflows)
    print('Average, std: {}, {}'.format(np.mean(final_inflows),
                                        np.std(final_inflows)))
    # Compute throughput efficiency in the last 500 sec of the
    print("Throughput efficiency (veh/hr):")
    print(throughput_efficiency)
    print('Average, std: {}, {}'.format(np.mean(throughput_efficiency),
                                        np.std(throughput_efficiency)))

    # terminate the environment
    env.unwrapped.terminate()

    # if prompted, convert the emission file into a csv file
    if args.gen_emission:
        time.sleep(0.1)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(env.network.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        # convert the emission file into a csv file
        emission_to_csv(emission_path)

        # print the location of the emission csv file
        emission_path_csv = emission_path[:-4] + ".csv"
        print("\nGenerated emission file at " + emission_path_csv)

        # delete the .xml version of the emission file
        os.remove(emission_path)


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        '--rllib_result_dir', '-r', required=False, type=str, help='Directory containing results')
    parser.add_argument('--checkpoint_num', '-c', required=False, type=str, help='Checkpoint number.')

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
        '--render_mode', '-rm',
        type=str,
        default=None,
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    parser.add_argument(
        '--local',
        action='store_true',
        help='Adjusts run settings to be compatible with limited '
             'memory capacity'
    )
    parser.add_argument(
        '--controller',
        type=str,
        help='Which custom controller to use. Defaults to IDM'
    )
    parser.add_argument(
        '--run_transfer',
        action='store_true',
        help='Runs transfer tests if true'
    )
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    rllib_config = None
    rllib_result_dir = None
    if args.rllib_result_dir is not None:
        rllib_result_dir = args.rllib_result_dir if args.rllib_result_dir[-1] != '/' \
            else args.rllib_result_dir[:-1]

        rllib_config = get_rllib_config(rllib_result_dir)

    flow_params = deepcopy(I210_MA_DEFAULT_FLOW_PARAMS)

    if args.local:
        ray.init(num_cpus=1, object_store_memory=200 * 1024 * 1024)
    else:
        ray.init(num_cpus=1)

    if args.run_transfer:
        for transfer_test in inflows_range(penetration_rates=[0.05, 0.1, 0.2], flow_rate_coefs=[0.8, 1.0, 1.2]):
            replay(args, flow_params, transfer_test, rllib_config=rllib_config, result_dir=rllib_result_dir)
    else:
        # def replay(args, flow_params, transfer_test, rllib_config=None, result_dir=None):
        replay(args, flow_params, rllib_config=rllib_config, result_dir=rllib_result_dir)
