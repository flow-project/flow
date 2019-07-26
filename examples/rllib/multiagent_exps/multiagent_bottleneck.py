"""Multi-agent Bottleneck example.
In this example, each agent is given a single acceleration per timestep.

The agents all share a single model.
"""
import argparse
from datetime import datetime
import json

import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController

# TODO(@evinitsky) clean this up
EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


def setup_rllib_params(args):
    # time horizon of a single rollout
    horizon = args.horizon
    # number of parallel workers
    n_cpus = args.n_cpus
    # number of rollouts per training iteration scaled by how many sets of rollouts per iter we want
    n_rollouts = args.n_cpus * args.rollout_scale_factor
    return {'horizon': horizon, 'n_cpus': n_cpus, 'n_rollouts': n_rollouts}


def setup_flow_params(args):
    DISABLE_TB = True
    DISABLE_RAMP_METER = True
    av_frac = args.av_frac
    if args.lc_on:
        lc_mode = 1621
    else:
        lc_mode = 0

    vehicles = VehicleParams()
    if not np.isclose(av_frac, 1):
        vehicles.add(
            veh_id="human",
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=9,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=lc_mode,
            ),
            num_vehicles=1)
        vehicles.add(
            veh_id="av",
            acceleration_controller=(RLController, {}),
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=9,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,
            ),
            num_vehicles=1)
    else:
        vehicles.add(
            veh_id="av",
            acceleration_controller=(RLController, {}),
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=9,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,
            ),
            num_vehicles=1)

    # flow rate
    flow_rate = 1900 * args.scaling

    controlled_segments = [('1', 1, False), ('2', 2, True), ('3', 2, True),
                           ('4', 2, True), ('5', 1, False)]
    num_observed_segments = [('1', 1), ('2', 3), ('3', 3), ('4', 3), ('5', 1)]
    additional_env_params = {
        'target_velocity': 40,
        'disable_tb': True,
        'disable_ramp_metering': True,
        'controlled_segments': controlled_segments,
        'symmetric': False,
        'observed_segments': num_observed_segments,
        'reset_inflow': True,
        'lane_change_duration': 5,
        'max_accel': 3,
        'max_decel': 3,
        'inflow_range': [800, 2000],
        'start_inflow': flow_rate,
        'congest_penalty': args.congest_penalty,
        'communicate': args.communicate,
        "centralized_obs": args.central_obs,
        "aggregate_info": args.aggregate_info,
        "av_frac": args.av_frac,
        "congest_penalty_start": args.congest_penalty_start,
        "lc_mode": lc_mode
    }

    # percentage of flow coming out of each lane
    inflow = InFlows()
    if not np.isclose(args.av_frac, 1.0):
        inflow.add(
            veh_type='human',
            edge='1',
            vehs_per_hour=flow_rate * (1 - args.av_frac),
            departLane='random',
            departSpeed=10.0)
        inflow.add(
            veh_type='av',
            edge='1',
            vehs_per_hour=flow_rate * args.av_frac,
            departLane='random',
            departSpeed=10.0)
    else:
        inflow.add(
            veh_type='av',
            edge='1',
            vehs_per_hour=flow_rate,
            departLane='random',
            departSpeed=10.0)

    traffic_lights = TrafficLightParams()
    if not DISABLE_TB:
        traffic_lights.add(node_id='2')
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id='3')

    additional_net_params = {'scaling': args.scaling, "speed_limit": 23.0}

    flow_params = dict(
        # name of the experiment
        exp_tag=args.exp_title,

        # name of the flow environment the experiment is running on
        env_name='MultiBottleneckEnv',

        # name of the scenario class the experiment is running on
        scenario='BottleneckScenario',

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.5,
            render=args.render,
            print_warnings=False,
            restart_instance=True,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            warmup_steps=40,
            sims_per_step=1,
            horizon=args.horizon,
            additional_params=additional_env_params,
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # scenario's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            no_internal_links=False,
            additional_params=additional_net_params,
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.vehicles.Vehicles)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing='uniform',
            min_gap=5,
            lanes_distribution=float('inf'),
            edges_distribution=['2', '3', '4', '5'],
        ),

        # traffic lights to be introduced to specific nodes (see
        # flow.core.traffic_lights.TrafficLights)
        tls=traffic_lights,
    )
    return flow_params


def setup_exps(args):
    rllib_params = setup_rllib_params(args)
    flow_params = setup_flow_params(args)
    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    config['num_workers'] = rllib_params['n_cpus']
    config['train_batch_size'] = args.horizon * rllib_params['n_rollouts']
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [64, 64]})
    config['clip_actions'] = False
    config['horizon'] = args.horizon
    # config['use_centralized_vf'] = tune.grid_search([True, False])
    # config['max_vf_agents'] = 140
    config['simple_optimizer'] = True
    # config['vf_clip_param'] = 100

    # Grid search things
    if args.grid_search:
        config['lr'] = tune.grid_search([5e-5, 5e-4])
        config['num_sgd_iter'] = tune.grid_search([10, 30])

    # LSTM Things
    config['model']['use_lstm'] = args.use_lstm
    if args.use_lstm:
        config['model']["max_seq_len"] = tune.grid_search([5, 10])
    config['model']["lstm_cell_size"] = 64

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    # Setup PG with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'av': (None, obs_space, act_space, {})}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            "policies_to_train": ["av"]
        }
    })
    return alg_run, env_name, config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Parses command line args for multi-agent bottleneck exps',
        epilog=EXAMPLE_USAGE)

    # required input parameters for tune
    parser.add_argument('exp_title', type=str, help='Informative experiment title to help distinguish results')
    parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
    parser.add_argument('--n_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--multi_node', type=bool, default=False, help='Set to true if this will '
                                                                       'be run in cluster mode')
    parser.add_argument("--num_iters", type=int, default=350)
    parser.add_argument("--checkpoint_freq", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--grid_search", type=bool, default=False)

    # arguments for flow
    parser.add_argument('--render', action='store_true', help='Show sumo-gui of results')
    parser.add_argument('--horizon', type=int, default=2000, help='Horizon of the environment')
    parser.add_argument('--av_frac', type=float, default=0.1, help='What fraction of the vehicles should be autonomous')
    parser.add_argument('--scaling', type=int, default=1, help='How many lane should we start with. Value of 1 -> 4, '
                                                               '2 -> 8, etc.')
    parser.add_argument('--lc_on', type=bool, default=False, help='If true, lane changing is enabled.')
    parser.add_argument('--congest_penalty', type=bool, default=False, help='If true, an additional penalty is added '
                                                                            'for vehicles queueing in the bottleneck')
    parser.add_argument('--communicate', type=bool, default=False, help='If true, the agents have an additional action '
                                                                        'which consists of sending a discrete signal '
                                                                        'to all nearby vehicles')
    parser.add_argument('--central_obs', type=bool, default=False, help='If true, all agents receive the same '
                                                                        'aggregate statistics')
    parser.add_argument('--aggregate_info', type=bool, default=False, help='If true, agents receive some '
                                                                           'centralized info')
    parser.add_argument('--congest_penalty_start', type=int, default=30, help='If congest_penalty is true, this '
                                                                              'sets the number of vehicles in edge 4'
                                                                              'at which the penalty sets in')

    # arguments for ray
    parser.add_argument('--rollout_scale_factor', type=int, default=1, help='the total number of rollouts is'
                                                                            'args.n_cpus * rollout_scale_factor')
    parser.add_argument('--use_lstm', type=bool, default=False)

    args = parser.parse_args()

    alg_run, env_name, config = setup_exps(args)
    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    else:
        ray.init(num_cpus=args.n_cpus + 1)
    s3_string = "s3://eugene.experiments/trb_bottleneck_paper/" \
                + datetime.now().strftime("%m-%d-%Y") + '/' + args.exp_title
    exp_dict = {
        args.exp_title: {
            'run': alg_run,
            'env': env_name,
            'checkpoint_freq': args.checkpoint_freq,
            'stop': {
                'training_iteration': args.num_iters
            },
            'config': config,
            'num_samples': args.num_samples,
        },
    }
    if args.use_s3:
        exp_dict[args.exp_title]['upload_dir'] = s3_string

    run_experiments(exp_dict)
