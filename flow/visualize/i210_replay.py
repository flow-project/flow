"""Transfer and replay for i210 environment."""
import argparse
from datetime import datetime
from copy import deepcopy
import os
import pytz
import subprocess

from examples.exp_configs.rl.multiagent.multiagent_i210 import flow_params as I210_MA_DEFAULT_FLOW_PARAMS
from examples.exp_configs.rl.multiagent.multiagent_i210 import custom_callables
from flow.core.experiment import Experiment
from flow.utils.registry import make_create_env
from flow.visualize.transfer.util import inflows_range
from flow.visualize.visualizer_rllib import read_result_dir, set_sim_params, set_env_params, set_agents, get_rl_action
import ray
from ray.tune.registry import register_env

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

    sim_params = set_sim_params(flow_params['sim'], args.render_mode, args.save_render)

    set_env_params(flow_params['env'], args.evaluate, args.horizon)

    # Create and register a gym+rllib env
    exp = Experiment(flow_params, custom_callables=custom_callables)

    if args.render_mode == 'sumo_gui':
        exp.env.sim_params.render = True  # set to True after initializing agent and env

    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        exp.env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    # reroute on exit is a training hack, it should be turned off at test time.
    if hasattr(exp.env, "reroute_on_exit"):
        exp.env.reroute_on_exit = False

    policy_map_fn, rets = None, None
    if rllib_config:
        result_dir, rllib_config, multiagent, rllib_flow_params = read_result_dir(rllib_config, True)

        # lower the horizon if testing
        if args.horizon:
            rllib_config['horizon'] = args.horizon

        agent_create_env, agent_env_name = make_create_env(params=rllib_flow_params, version=0)
        register_env(agent_env_name, agent_create_env)

        assert 'run' in rllib_config['env_config'], "Was this trained with the latest version of Flow?"
        # Determine agent and checkpoint
        agent = set_agents(rllib_config, result_dir, agent_env_name)

        rllib_rl_action, policy_map_fn, rets = get_rl_action(rllib_config, agent, multiagent)

    # reroute on exit is a training hack, it should be turned off at test time.
    if hasattr(exp.env, "reroute_on_exit"):
        exp.env.reroute_on_exit = False

    def rl_action(state):
        if rllib_config:
            action = rllib_rl_action(state)
        else:
            action = None
        return action

    info_dict = exp.run(num_runs=args.num_rollouts, convert_to_csv=args.gen_emission, to_aws=args.use_s3,
                        rl_actions=rl_action, multiagent=rllib_config and multiagent, rets=rets,
                        policy_map_fn=policy_map_fn)

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
    parser.add_argument(
        '--is_baseline',
        action='store_true',
        help='specifies whether this is a baseline run'
    )
    parser.add_argument(
        '--exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/non_rl.')
    return parser


if __name__ == '__main__':
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")

    parser = create_parser()
    args = parser.parse_args()

    if args.exp_config:
        module = __import__("../../examples/exp_configs.non_rl", fromlist=[args.exp_config])
        flow_params = getattr(module, args.exp_config).flow_params
    else:
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
                                    rllib_config=args.rllib_result_dir, result_dir=args.rllib_result_dir,
                                    max_completed_trips=args.max_completed_trips)
                      for transfer_test in s]
        ray.get(ray_output)

    elif args.v_des_sweep:
        assert args.controller == 'follower_stopper'

        ray_output = [
            replay.remote(args, flow_params, output_dir="{}/{}".format(output_dir, v_des),
                          rllib_config=args.rllib_result_dir, result_dir=args.rllib_result_dir,
                          max_completed_trips=args.max_completed_trips, v_des=v_des)
            for v_des in range(8, 17, 2)]
        ray.get(ray_output)

    else:
        if args.penetration_rate is not None:
            pr = args.penetration_rate if args.penetration_rate is not None else 0
            single_transfer = next(inflows_range(penetration_rates=pr))
            ray.get(replay.remote(args, flow_params, output_dir=output_dir, transfer_test=single_transfer,
                                  rllib_config=args.rllib_result_dir, result_dir=args.rllib_result_dir,
                                  max_completed_trips=args.max_completed_trips))
        else:
            ray.get(replay.remote(args, flow_params, output_dir=output_dir,
                                  rllib_config=args.rllib_result_dir, result_dir=args.rllib_result_dir,
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
