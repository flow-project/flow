"""Transfer and replay for i210 environment."""
import argparse
import numpy as np
import os
from copy import deepcopy

from flow.controllers.car_following_models import IDMController, SimCarFollowingController
from flow.controllers.rlcontroller import RLController
from flow.controllers.velocity_controllers import FollowerStopper

from examples.exp_configs.rl.multiagent.multiagent_i210 import flow_params as I210_MA_DEFAULT_FLOW_PARAMS
from examples.exp_configs.rl.multiagent.multiagent_i210 import custom_callables
from flow.core.experiment import Experiment
from flow.envs.multiagent.i210 import I210TransferEnv
from flow.utils.registry import make_create_env
from flow.visualize.transfer.util import inflows_range
from flow.replay.rllib_replay import read_result_dir
from flow.replay.rllib_replay import set_sim_params
from flow.replay.rllib_replay import set_env_params
from flow.replay.rllib_replay import set_agents
from flow.replay.rllib_replay import get_rl_action
import ray
from ray.tune.registry import register_env

from flow.data_pipeline.data_pipeline import collect_metadata_from_config

EXAMPLE_USAGE = """
example usage:
    python transfer_tests.py -r /ray_results/experiment_dir/result_dir -c 1
    python transfer_tests.py --controller idm
    python transfer_tests.py --controller idm --run_transfer

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


@ray.remote
def replay(args,
           flow_params,
           output_dir=None,
           transfer_test=None,
           rllib_config=None,
           max_completed_trips=None,
           supplied_metadata=None,
           inflow_rate=None,
           outflow_speed_limit=None,
           controller_params={},
           default_controller_params={}):
    """Replay or run transfer test (defined by transfer_fn) by modif.

    Parameters
    ----------
    args : argparse.Namespace
        input arguments passed via a parser. See create_parser.
    flow_params : dict
        flow-specific parameters
    output_dir : str
        Directory to save results.
    transfer_test : TODO
        TODO
    rllib_config : str
        Directory containing rllib results
    max_completed_trips : int
        Terminate rollout after max_completed_trips vehicles have started and
        ended.
    supplied_metadata: dict (str : list)
        the metadata associated with this simulation.
    inflow_rate : int
        The inflow rate. Analogous to what you see in multiagent_i210's exp_config
    outflow_speed_limit : int
        The speed limit of the final edge
    controller_params : dict
        Custom params for the once-RL vehicle. e.g. If you replace the RL vehicles with
        IDM vehicles, this could be IDM params
    default_controller_params : dict
        Custom params for the once-human vehicle e.g. If you replace the human vehicles with
        IDM vehicles, this could be IDM params
    """
    args.gen_emission = args.gen_emission or args.use_s3

    if transfer_test is not None:
        if type(transfer_test) == bytes:
            transfer_test = ray.cloudpickle.loads(transfer_test)
        flow_params = transfer_test.flow_params_modifier_fn(flow_params)

    # Choose the controller to replace the AV, as well as initialize with any
    # controller-specific parameters
    if args.controller:
        test_params = {}
        if args.controller == 'idm':
            controller = IDMController
            if 'idm_params' in controller_params:
                test_params.update(controller_params['idm_params'])
        elif args.controller == 'default_human':
            controller = flow_params['veh'].type_parameters['human']['acceleration_controller'][0]
            test_params.update(flow_params['veh'].type_parameters['human']['acceleration_controller'][1])
        elif args.controller == 'follower_stopper':
            if 'v_des' in controller_params:
                test_params.update({'v_des': controller_params['v_des']})
            else:
                test_params.update({'v_des': 12})
        elif args.controller == 'sumo':
            controller = SimCarFollowingController

        # Update the parameters
        flow_params['veh'].type_parameters['rl']['acceleration_controller'] = (controller, test_params)
        for veh_param in flow_params['veh'].initial:
            if veh_param['veh_id'] == 'rl':
                veh_param['acceleration_controller'] = (controller, test_params)

    # Choose the controller to replace the original human controller, as well as initialize with any
    # controller-specific parameters
    if args.default_controller:
        test_params = {}
        if args.default_controller == 'idm':
            default_controller = IDMController
            # If the original human-controlled vehicle was IDM, first initialize with its
            # parameters before applying other modifications.
            if flow_params['veh'].type_parameters['human']['acceleration_controller'][0] == IDMController:
                test_params.update(flow_params['veh'].type_parameters['human']['acceleration_controller'][1])
            if 'idm_params' in default_controller_params:
                test_params.update(default_controller_params['idm_params'])
        elif args.default_controller == 'follower_stopper':
            default_controller = FollowerStopper
            # If the original human-controlled vehicle was FollowerStopper, first initialize with its
            # parameters before applying other modifications.
            if flow_params['veh'].type_parameters['human']['acceleration_controller'][0] == FollowerStopper:
                test_params.update(flow_params['veh'].type_parameters['human']['acceleration_controller'][1])
            if 'v_des' in default_controller_params:
                test_params.update({'v_des': default_controller_params['v_des']})
            else:
                test_params.update({'v_des': 12})
        elif args.default_controller == 'sumo':
            default_controller = SimCarFollowingController

        # Update the parameters
        flow_params['veh'].type_parameters['human']['acceleration_controller'] = (default_controller, test_params)
        for veh_param in flow_params['veh'].initial:
            if veh_param['veh_id'] == 'human':
                veh_param['acceleration_controller'] = (default_controller, test_params)

    # Updates the values of the default IDM controller applied to once-human vehicles.
    if args.idm_sweep:
        test_params = {}
        test_params.update(flow_params['veh'].type_parameters['human']['acceleration_controller'][1])
        test_params.update(default_controller_params['idm_params'])
        default_controller = IDMController

        # Update the parameters
        flow_params['veh'].type_parameters['human']['acceleration_controller'] = (default_controller, test_params)
        for veh_param in flow_params['veh'].initial:
            if veh_param['veh_id'] == 'human':
                veh_param['acceleration_controller'] = (controller, test_params)

    # Modifies the lcSpeedGain parameter of once-human vehicles
    if args.lane_freq_sweep:
        test_lane_params = {'lcSpeedGain': str(float(default_controller_params['lane_frequency']))}

        flow_params['veh'].type_parameters['human']['lane_change_params'].controller_params.update(test_lane_params)
        for veh_param in flow_params['veh'].initial:
            if veh_param['veh_id'] == 'human':
                veh_param['lane_change_params'].controller_params.update(test_lane_params)
        for vtype in flow_params['veh'].types:
            if vtype['veh_id'] == 'human':
                vtype['type_params']['lcSpeedGain'] = str(float(default_controller_params['lane_frequency']))

    # TODO is there a more dynamic way instead of hardcoding the start edges
    # Modifies the inflow rate
    if args.inflow_sweep:
        from examples.exp_configs.rl.multiagent.multiagent_i210 import PENETRATION_RATE, \
            ON_RAMP, WANT_BOUNDARY_CONDITIONS, ON_RAMP_INFLOW_RATE, ENTER_AS_LINE
        for inflow in flow_params['net'].inflows.get():
            if ENTER_AS_LINE:
                if WANT_BOUNDARY_CONDITIONS:
                    if inflow['edge'] == 'ghost0':
                        if inflow['vtype'] == 'human':
                            inflow['vehsPerHour'] = int(inflow_rate * (1 - PENETRATION_RATE))
                        elif inflow['vtype'] == 'rl':
                            inflow['vehsPerHour'] = int(inflow_rate * PENETRATION_RATE)
                else:
                    if inflow['edge'] == "119257914":
                        if inflow['vtype'] == 'human':
                            inflow['vehsPerHour'] = int(inflow_rate * (1 - PENETRATION_RATE))
                        elif inflow['vtype'] == 'rl':
                            inflow['vehsPerHour'] = int(inflow_rate * PENETRATION_RATE)
                if ON_RAMP:
                    if inflow['edge'] == '27414345' or inflow['edge'] == "27414342#0":
                        inflow['vehsPerHour'] = int(ON_RAMP_INFLOW_RATE * (1 - PENETRATION_RATE))

            else:
                if WANT_BOUNDARY_CONDITIONS:
                    if inflow['edge'] == 'ghost0':
                        if inflow['vtype'] == 'human':
                            inflow['vehsPerHour'] = int(inflow_rate * 5 * (1 - PENETRATION_RATE))
                        elif inflow['vtype'] == 'rl':
                            inflow['vehsPerHour'] = int(inflow_rate * 5 * PENETRATION_RATE)
                else:
                    if inflow['edge'] == '119257914':
                        if inflow['vtype'] == 'human':
                            inflow['vehsPerHour'] = int(inflow_rate * 5 * (1 - PENETRATION_RATE))
                        elif inflow['vtype'] == 'rl':
                            inflow['vehsPerHour'] = int(inflow_rate * 5 * PENETRATION_RATE)

                if ON_RAMP:
                    if inflow['edge'] == '27414345' or inflow['edge'] == '27414342#0':
                        inflow['vehsPerHour'] = int(ON_RAMP_INFLOW_RATE * (1 - PENETRATION_RATE))

    # Modifies the speed limit of the last edge
    if args.outflow_sweep:
        flow_params['env'].additional_params['outflow_speed_limit'] = outflow_speed_limit

    set_sim_params(flow_params['sim'], args.render_mode, args.save_render, args.gen_emission, output_dir)
    sim_params = flow_params['sim']

    set_env_params(flow_params['env'], args.evaluate, args.horizon)

    # Create and register a gym+rllib env
    exp = Experiment(flow_params, custom_callables=custom_callables)

    # set to True after initializing agent and env
    if args.render_mode == 'sumo_gui':
        exp.env.sim_params.render = True

    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        exp.env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    # reroute on exit is a training hack, it should be turned off at test time.
    if hasattr(exp.env, "reroute_on_exit"):
        exp.env.reroute_on_exit = False

    policy_map_fn, rets = None, None
    if rllib_config:
        result_dir, rllib_config, multiagent, rllib_flow_params = read_result_dir(rllib_config)

        # lower the horizon if testing
        if args.horizon:
            rllib_config['horizon'] = args.horizon

        agent_create_env, agent_env_name = make_create_env(params=rllib_flow_params, version=0)
        register_env(agent_env_name, agent_create_env)

        assert 'run' in rllib_config['env_config'], \
            "Was this trained with the latest version of Flow?"
        # Determine agent and checkpoint
        agent = set_agents(rllib_config, result_dir, agent_env_name, checkpoint_num=args.checkpoint_num)

        policy_map_fn, rllib_rl_action, rets = get_rl_action(rllib_config, agent, multiagent)

    # reroute on exit is a training hack, it should be turned off at test time.
    if hasattr(exp.env, "reroute_on_exit"):
        exp.env.reroute_on_exit = False

    def rl_action(state):
        if rllib_config:
            action = rllib_rl_action(state)
        else:
            action = None
        return action

    info_dict = exp.run(
        num_runs=args.num_rollouts,
        convert_to_csv=args.gen_emission,
        to_aws=args.use_s3,
        rl_actions=rl_action,
        multiagent=True,
        rets=rets,
        policy_map_fn=policy_map_fn,
        supplied_metadata=supplied_metadata
    )

    return info_dict


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        '--rllib_result_dir', '-r',
        required=False,
        type=str,
        help='Directory containing results'
    )
    parser.add_argument(
        '--checkpoint_num', '-c',
        required=False,
        type=str,
        default=None,
        help='Checkpoint number.'
    )
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
        '--default_controller',
        type=str,
        help='Which controller the non-controlled vehicles use. Defaults to IDM'
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
        help='Terminate rollout after max_completed_trips vehicles have '
             'started and ended.',
        default=None)
    parser.add_argument(
        '--v_des_sweep',
        action='store_true',
        help='Runs a sweep over v_des params.',
        default=None)
    parser.add_argument(
        '--idm_sweep',
        action='store_true',
        help='Runs a sweep over idm params.',
        default=None)
    parser.add_argument(
        '--inflow_sweep',
        action='store_true',
        help='Runs a sweep over the inflows of other vehicles.',
        default=None)
    parser.add_argument(
        '--outflow_sweep',
        action='store_true',
        help='Runs a sweep over the outflow speed.',
        default=None)
    parser.add_argument(
        '--lane_freq_sweep',
        action='store_true',
        help='Runs a sweep over the lane change frequencies.',
        default=None)
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Directory to save results.',
        default=None
    )
    parser.add_argument(
        '--use_s3',
        action='store_true',
        help='If true, upload results to s3'
    )
    parser.add_argument(
        '--num_cpus',
        type=int,
        default=1,
        help='Number of cpus to run experiment with'
    )
    parser.add_argument(
        '--multi_node',
        action='store_true',
        help='Set to true if this will be run in cluster mode'
    )
    parser.add_argument(
        '--exp_title',
        type=str,
        required=False,
        default=None,
        help='Informative experiment title to help distinguish results'
    )
    parser.add_argument(
        '--only_query',
        nargs='*', default="[\'all\']",
        help='specify which query should be run by lambda for detail, see '
             'upload_to_s3 in data_pipeline.py'
    )
    parser.add_argument(
        '--is_baseline',
        action='store_true',
        help='specifies whether this is a baseline run'
    )
    parser.add_argument(
        '--no_warmup',
        action='store_true',
        help='Set warmup steps to 0. Mostly for debugging purposes'
    )
    parser.add_argument(
        '--no_warnings',
        action='store_true',
        help='Sets display_warnings to False in the vehicle definition'
    )
    parser.add_argument(
        '--exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/non_rl.')
    parser.add_argument(
        '--submitter_name', type=str,
        help='Name displayed next to the submission on the leaderboard.')
    parser.add_argument(
        '--strategy_name', type=str,
        help='Strategy displayed next to the submission on the leaderboard.')
    return parser


def generate_graphs(args):
    """Run the replay according to the commandline arguments."""
    supplied_metadata = None
    if args.submitter_name and args.strategy_name:
        supplied_metadata = {'name': args.submitter_name,
                             'strategy': args.strategy_name}
    if args.exp_config:
        module = __import__("../../examples/exp_configs.non_rl", fromlist=[args.exp_config])
        flow_params = getattr(module, args.exp_config).flow_params
        supplied_metadata = collect_metadata_from_config(getattr(module, args.exp_config))
    else:
        flow_params = deepcopy(I210_MA_DEFAULT_FLOW_PARAMS)
    # Use the TestEnv
    flow_params['env_name'] = I210TransferEnv
    if args.no_warmup:
        flow_params['env'].warmup_steps = 0

    if args.no_warnings:
        for type_param in flow_params['veh'].type_parameters.values():
            if not type_param['acceleration_controller'][0] == RLController:
                type_param['acceleration_controller'][1]['display_warnings'] = False
        for veh_param in flow_params['veh'].initial:
            if not type_param['acceleration_controller'][0] == RLController:
                veh_param['acceleration_controller'][1]['display_warnings'] = False

    if ray.is_initialized():
        ray.shutdown()
    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    elif args.local:
        ray.init(local_mode=True, object_store_memory=200 * 1024 * 1024)
    else:
        ray.init(num_cpus=args.num_cpus, object_store_memory=200 * 1024 * 1024)

    if args.exp_title:
        output_dir = os.path.join(args.output_dir, args.exp_title)
    else:
        output_dir = args.output_dir

    if args.run_transfer:
        s = [ray.cloudpickle.dumps(transfer_test) for transfer_test in
             inflows_range(penetration_rates=[0.0, 0.1, 0.2, 0.3])]
        ray_output = [
            replay.remote(
                args,
                flow_params,
                output_dir=output_dir,
                transfer_test=transfer_test,
                rllib_config=args.rllib_result_dir,
                max_completed_trips=args.max_completed_trips,
                supplied_metadata=supplied_metadata
            )
            for transfer_test in s
        ]
        ray.get(ray_output)

    elif args.v_des_sweep:
        assert args.controller == 'follower_stopper'

        ray_output = [
            replay.remote(
                args,
                flow_params,
                output_dir="{}/{}".format(output_dir, v_des),
                rllib_config=args.rllib_result_dir,
                max_completed_trips=args.max_completed_trips,
                supplied_metadata=supplied_metadata,
                controller_params={'v_des': v_des}
            )
            for v_des in range(8, 13, 1)
        ]
        ray.get(ray_output)

    elif args.idm_sweep:
        assert args.controller == 'idm'
        ray_output = [
            replay.remote(
                args,
                flow_params,
                output_dir="{}/{}".format(output_dir, a),
                rllib_config=args.rllib_result_dir,
                max_completed_trips=args.max_completed_trips,
                default_controller_params={'idm_params': {'a': a}}
            )
            for a in np.arange(0.2, 2, 0.2)]
        ray.get(ray_output)

    elif args.inflow_sweep:
        ray_output = [
            replay.remote(
                args,
                flow_params,
                output_dir="{}/{}".format(output_dir, inflow_rate),
                rllib_config=args.rllib_result_dir,
                max_completed_trips=args.max_completed_trips,
                inflow_rate=inflow_rate
            )
            for inflow_rate in range(1800, 2500, 100)]
        ray.get(ray_output)

    elif args.outflow_sweep:
        ray_output = [
            replay.remote(
                args,
                flow_params,
                output_dir="{}/{}".format(output_dir, outflow_speed_limit),
                rllib_config=args.rllib_result_dir,
                max_completed_trips=args.max_completed_trips,
                outflow_speed_limit=outflow_speed_limit
            )
            for outflow_speed_limit in range(2, 10, 2)]
        ray.get(ray_output)

    elif args.lane_freq_sweep:
        ray_output = [
            replay.remote(
                args,
                flow_params,
                output_dir="{}/{}".format(output_dir, lf),
                rllib_config=args.rllib_result_dir,
                max_completed_trips=args.max_completed_trips,
                default_controller_params={'lane_frequency': lf}
            )
            for lf in [1, 10, 20]]
        ray.get(ray_output)

    else:
        if args.penetration_rate is not None:
            pr = args.penetration_rate
            single_transfer = next(inflows_range(penetration_rates=pr))
            ray.get(replay.remote(
                args,
                flow_params,
                output_dir=output_dir,
                transfer_test=single_transfer,
                rllib_config=args.rllib_result_dir,
                max_completed_trips=args.max_completed_trips,
                supplied_metadata=supplied_metadata
            ))
        else:
            ray.get(replay.remote(
                args,
                flow_params,
                output_dir=output_dir,
                rllib_config=args.rllib_result_dir,
                max_completed_trips=args.max_completed_trips,
                supplied_metadata=supplied_metadata
            ))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    generate_graphs(args)
