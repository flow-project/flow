"""Transfer and replay for i210 environment."""
import argparse
from datetime import datetime
from collections import defaultdict
from copy import deepcopy
import numpy as np
import json
import os
import pytz
import subprocess
import time

import ray

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv, ensure_dir
from flow.core.rewards import vehicle_energy_consumption
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl
from flow.utils.rllib import FlowParamsEncoder

from flow.visualize.transfer.util import inflows_range
from flow.visualize.plot_custom_callables import plot_trip_distribution

from examples.exp_configs.rl.multiagent.multiagent_i210 import flow_params as I210_MA_DEFAULT_FLOW_PARAMS
from examples.exp_configs.rl.multiagent.multiagent_i210 import custom_callables

from flow.data_pipeline.data_pipeline import generate_trajectory_from_flow, upload_to_s3, extra_init, get_extra_info
import uuid

EXAMPLE_USAGE = """
example usage:
    python i210_replay.py -r /ray_results/experiment_dir/result_dir -c 1
    python i210_replay.py --controller idm
    python i210_replay.py --controller idm --run_transfer

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


@ray.remote
def replay(args, flow_params, output_dir=None, transfer_test=None, rllib_config=None, result_dir=None,
           max_completed_trips=None, v_des=12):
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
    if transfer_test is not None:
        if type(transfer_test) == bytes:
            transfer_test = ray.cloudpickle.loads(transfer_test)
        flow_params = transfer_test.flow_params_modifier_fn(flow_params)

    if args.controller:
        test_params = {}
        if args.controller == 'idm':
            from flow.controllers.car_following_models import IDMController
            controller = IDMController
            test_params.update({'v0': 1, 'T': 1, 'a': 0.2, 'b': 0.2})  # An example of really obvious changes
        elif args.controller == 'default_human':
            controller = flow_params['veh'].type_parameters['human']['acceleration_controller'][0]
            test_params.update(flow_params['veh'].type_parameters['human']['acceleration_controller'][1])
        elif args.controller == 'follower_stopper':
            from flow.controllers.velocity_controllers import FollowerStopper
            controller = FollowerStopper
            test_params.update({'v_des': v_des})
            # flow_params['veh'].type_parameters['av']['car_following_params']
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
            # map the agent id to its policy
            policy_map_fn = rllib_config['multiagent']['policy_mapping_fn']

        if rllib_config['model']['use_lstm']:
            use_lstm = True
            if multiagent:
                # map the agent id to its policy
                size = rllib_config['model']['lstm_cell_size']
                lstm_state = defaultdict(lambda: [np.zeros(size, np.float32),
                                                  np.zeros(size, np.float32)])
            else:
                lstm_state = [
                    np.zeros(rllib_config['model']['lstm_cell_size'], np.float32),
                    np.zeros(rllib_config['model']['lstm_cell_size'], np.float32)
                ]
        else:
            use_lstm = False

    # used to store
    info_dict = {
        "velocities": [],
        "outflows": [],
        "avg_trip_energy": [],
        "avg_trip_time": [],
        "total_completed_trips": []
    }
    all_trip_energy_distribution = defaultdict(lambda: [])
    all_trip_time_distribution = defaultdict(lambda: [])

    info_dict.update({
        key: [] for key in custom_callables.keys()
    })

    extra_info = extra_init()
    source_id = 'flow_{}'.format(uuid.uuid4().hex)

    i = 0
    while i < args.num_rollouts:
        print("Rollout iter", i)
        vel = []
        per_vehicle_energy_trace = defaultdict(lambda: [])
        completed_veh_types = {}
        completed_vehicle_avg_energy = {}
        completed_vehicle_travel_time = {}
        custom_vals = {key: [] for key in custom_callables.keys()}
        state = env.reset()
        initial_vehicles = set(env.k.vehicle.get_ids())
        for _ in range(env_params.horizon):
            if rllib_config:
                if multiagent:
                    action = {}
                    for agent_id in state.keys():
                        if use_lstm:
                            action[agent_id], lstm_state[agent_id], _ = \
                                agent.compute_action(
                                    state[agent_id], state=lstm_state[agent_id],
                                    policy_id=policy_map_fn(agent_id))
                        else:
                            action[agent_id] = agent.compute_action(
                                state[agent_id], policy_id=policy_map_fn(agent_id))
                else:
                    if use_lstm:
                        raise NotImplementedError
                    else:
                        action = agent.compute_action(state)
            else:
                action = None

            state, reward, done, _ = env.step(action)

            # Compute the velocity speeds and cumulative returns.
            veh_ids = env.k.vehicle.get_ids()
            vel.append(np.mean(env.k.vehicle.get_speed(veh_ids)))

            # Collect information from flow for the trajectory output
            get_extra_info(env.k.vehicle, extra_info, veh_ids)
            extra_info["source_id"].extend(['{}_run_{}'.format(source_id, i)] * len(veh_ids))

            # Compute the results for the custom callables.
            for (key, lambda_func) in custom_callables.items():
                custom_vals[key].append(lambda_func(env))

            for past_veh_id in per_vehicle_energy_trace.keys():
                if past_veh_id not in veh_ids and past_veh_id not in completed_vehicle_avg_energy:
                    all_trip_energy_distribution[completed_veh_types[past_veh_id]].append(
                        np.sum(per_vehicle_energy_trace[past_veh_id]))
                    all_trip_time_distribution[completed_veh_types[past_veh_id]].append(
                        len(per_vehicle_energy_trace[past_veh_id]))
                    completed_vehicle_avg_energy[past_veh_id] = np.sum(per_vehicle_energy_trace[past_veh_id])
                    completed_vehicle_travel_time[past_veh_id] = len(per_vehicle_energy_trace[past_veh_id])

            for veh_id in veh_ids:
                if veh_id not in initial_vehicles:
                    if veh_id not in per_vehicle_energy_trace:
                        # we have to skip the first step's energy calculation
                        per_vehicle_energy_trace[veh_id].append(0)
                        completed_veh_types[veh_id] = env.k.vehicle.get_type(veh_id)
                    else:
                        per_vehicle_energy_trace[veh_id].append(-1 * vehicle_energy_consumption(env, veh_id))

            if type(done) is dict and done['__all__']:
                break
            elif type(done) is not dict and done:
                break
            elif max_completed_trips is not None and len(completed_vehicle_avg_energy) > max_completed_trips:
                break
        if env.crash:
            print("Crash on iter", i)
        else:
            # Store the information from the run in info_dict.
            outflow = env.k.vehicle.get_outflow_rate(int(500))
            info_dict["velocities"].append(np.mean(vel))
            info_dict["outflows"].append(outflow)
            info_dict["avg_trip_energy"].append(np.mean(list(completed_vehicle_avg_energy.values())))
            info_dict["avg_trip_time"].append(np.mean(list(completed_vehicle_travel_time.values())))
            info_dict["total_completed_trips"].append(len(list(completed_vehicle_avg_energy.values())))
            for key in custom_vals.keys():
                info_dict[key].append(np.mean(custom_vals[key]))
            i += 1

    print('======== Summary of results ========')
    if args.run_transfer:
        print("Transfer test: {}".format(transfer_test.transfer_str))
    print("====================================")

    # Print the averages/std for all variables in the info_dict.
    for key in info_dict.keys():
        print("Average, std {}: {}, {}".format(
            key, np.mean(info_dict[key]), np.std(info_dict[key])))

    # terminate the environment
    env.unwrapped.terminate()

    if output_dir:
        ensure_dir(output_dir)
        if args.run_transfer:
            exp_name = "{}-replay".format(transfer_test.transfer_str)
        else:
            exp_name = "i210_replay"
        replay_out = os.path.join(output_dir, '{}-info.npy'.format(exp_name))
        np.save(replay_out, info_dict)
        # if prompted, convert the emission file into a csv file
        if args.gen_emission:
            emission_filename = '{0}-emission.xml'.format(env.network.name)
            time.sleep(0.1)

            emission_path = \
                '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

            output_path = os.path.join(output_dir, '{}-emission.csv'.format(exp_name))
            # convert the emission file into a csv file
            emission_to_csv(emission_path, output_path=output_path)

            # generate the trajectory output file
            trajectory_table_path = './data/' + source_id + ".csv"
            upload_file_path = generate_trajectory_from_flow(trajectory_table_path, extra_info)

            # upload to s3 if asked
            if args.use_s3:
                partition_name = date.today().isoformat() + " " + source_id[0:3]
                upload_to_s3('circles.data.pipeline', 'trajectory-output/' + 'partition_name=' + partition_name + '/'
                             + upload_file_path.split('/')[-1].split('_')[0] + '.csv',
                             upload_file_path, str(args.only_query)[2:-2])

            # print the location of the emission csv file
            print("\nGenerated emission file at " + output_path)

            # delete the .xml version of the emission file
            os.remove(emission_path)

        all_trip_energies = os.path.join(output_dir, '{}-all_trip_energies.npy'.format(exp_name))
        np.save(all_trip_energies, dict(all_trip_energy_distribution))
        fig_names, figs = plot_trip_distribution(all_trip_energy_distribution)

        for fig_name, fig in zip(fig_names, figs):
            edist_out = os.path.join(output_dir, '{}_energy_distribution.png'.format(fig_name))
            fig.savefig(edist_out)

        # Create the flow_params object
        with open(os.path.join(output_dir, exp_name) + '.json', 'w') as outfile:
            json.dump(flow_params, outfile,
                      cls=FlowParamsEncoder, sort_keys=True, indent=4)

    return info_dict


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
    parser.add_argument(
        '-pr',
        '--penetration_rate',
        type=float,
        help='Specifies percentage of AVs.',
        required=False)
    parser.add_argument(
        '-mct',
        '--max_completed_trips',
        type=int,
        help='Terminate rollout after max_completed_trips vehicles have started and ended.',
        default=None)
    parser.add_argument(
        '--v_des_sweep',
        action='store_true',
        help='Runs a sweep over v_des params.',
        default=None)
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Directory to save results.',
        default=None
    )
    parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--multi_node', action='store_true', help='Set to true if this will '
                                                                  'be run in cluster mode')
    parser.add_argument('--exp_title', type=str, required=False, default=None,
                        help='Informative experiment title to help distinguish results')
    parser.add_argument(
        '--only_query',
        nargs='*', default="[\'all\']",
        help='specify which query should be run by lambda'
             'for detail, see upload_to_s3 in data_pipeline.py'
    )
    return parser


if __name__ == '__main__':
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")

    parser = create_parser()
    args = parser.parse_args()

    rllib_config = None
    rllib_result_dir = None
    if args.rllib_result_dir is not None:
        rllib_result_dir = args.rllib_result_dir if args.rllib_result_dir[-1] != '/' \
            else args.rllib_result_dir[:-1]

        rllib_config = get_rllib_config(rllib_result_dir)

    flow_params = deepcopy(I210_MA_DEFAULT_FLOW_PARAMS)

    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    elif args.local:
        ray.init(local_mode=True, object_store_memory=200 * 1024 * 1024)
    else:
        ray.init(num_cpus=args.num_cpus + 1, object_store_memory=200 * 1024 * 1024)

    if args.exp_title:
        output_dir = os.path.join(args.output_dir, args.exp_title)
    else:
        output_dir = args.output_dir

    if args.run_transfer:
        s = [ray.cloudpickle.dumps(transfer_test) for transfer_test in
             inflows_range(penetration_rates=[0.0, 0.1, 0.2, 0.3])]
        ray_output = [replay.remote(args, flow_params, output_dir=output_dir, transfer_test=transfer_test,
                                    rllib_config=rllib_config, result_dir=rllib_result_dir,
                                    max_completed_trips=args.max_completed_trips)
                      for transfer_test in s]
        ray.get(ray_output)

    elif args.v_des_sweep:
        assert args.controller == 'follower_stopper'

        ray_output = [
            replay.remote(args, flow_params, output_dir="{}/{}".format(output_dir, v_des), rllib_config=rllib_config,
                          result_dir=rllib_result_dir, max_completed_trips=args.max_completed_trips, v_des=v_des)
            for v_des in range(8, 17, 2)]
        ray.get(ray_output)

    else:
        if args.penetration_rate is not None:
            pr = args.penetration_rate if args.penetration_rate is not None else 0
            single_transfer = next(inflows_range(penetration_rates=pr))
            ray.get(replay.remote(args, flow_params, output_dir=output_dir, transfer_test=single_transfer,
                                  rllib_config=rllib_config, result_dir=rllib_result_dir,
                                  max_completed_trips=args.max_completed_trips))
        else:
            ray.get(replay.remote(args, flow_params, output_dir=output_dir,
                                  rllib_config=rllib_config, result_dir=rllib_result_dir,
                                  max_completed_trips=args.max_completed_trips))

    if args.use_s3:
        s3_string = 's3://kanaad.experiments/i210_replay/' + date
        if args.exp_title:
            s3_string += '/' + args.exp_title

        for i in range(4):
            try:
                p1 = subprocess.Popen("aws s3 sync {} {}".format(output_dir, s3_string).split(' '))
                p1.wait(50)
            except Exception as e:
                print('This is the error ', e)
