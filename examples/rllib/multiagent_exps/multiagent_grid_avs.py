"""Grid with AVs example."""

import json
import argparse

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray import tune
from ray.tune.registry import register_env
from ray.tune import run_experiments

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import RLController
from flow.controllers import SimCarFollowingController, GridRouter

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder


def setup_exps_PPO(flow_params, n_rollouts, n_cpus, horizon):
    """
    Experiment setup with PPO using RLlib.

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = min(n_cpus, n_rollouts)
    config['train_batch_size'] = horizon * n_rollouts
    config['simple_optimizer'] = True
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [32, 32]})
    config['lr'] = tune.grid_search([1e-5])
    config['horizon'] = HORIZON
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config['observation_filter'] = 'NoFilter'

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

    def gen_policy():
        return (PPOPolicyGraph, obs_space, act_space, {})

    # Setup PG with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'av': gen_policy()}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policy_graphs': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            'policies_to_train': ['av']
        }
    })

    return alg_run, env_name, config


if __name__ == '__main__':
    EXAMPLE_USAGE = """
    example usage:
        python multiagent_traffic_light_grid_avs.py --upload_dir=<S3 bucket>
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="[Flow] Evaluates a multi-agent grid with AVs",
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument("--upload_dir", type=str,
                        help="S3 Bucket for uploading results.")

    # optional input parameters
    parser.add_argument('--run_mode', type=str, default='local',
                        help="Experiment run mode (local | cluster)")
    parser.add_argument('--algo', type=str, default='PPO',
                        help="RL method to use (PPO | ES)")
    parser.add_argument('--num_rows', type=int, default=1,
                        help="The number of rows in the grid network.")
    parser.add_argument('--num_cols', type=int, default=1,
                        help="The number of columns in the grid network.")
    parser.add_argument('--inflow_rate', type=int, default=600,
                        help="The inflow rate (veh/hr) per edge.")
    args = parser.parse_args()

    # Experiment parameters
    RUN_MODE = args.run_mode
    if RUN_MODE == 'local':
        N_ROLLOUTS = 1  # number of rollouts per training iteration
        N_CPUS = 1  # number of parallel workers
    elif RUN_MODE == 'cluster':
        N_ROLLOUTS = 63  # number of rollouts per training iteration
        N_CPUS = 63  # number of parallel workers

    # Environment parameters
    HORIZON = 400  # time horizon of a single rollout
    EDGE_INFLOW = args.inflow_rate  # inflow rate of vehicles at every edge
    V_ENTER = 30  # enter speed for departing vehicles
    N_ROWS = args.num_rows  # number of row of bidirectional lanes
    N_COLUMNS = args.num_cols  # number of columns of bidirectional lanes
    INNER_LENGTH = 300  # length of inner edges in the grid network
    LONG_LENGTH = 100  # length of final edge in route
    SHORT_LENGTH = 300  # length of edges that vehicles start on
    # number of vehicles originating in the left, right, top, and bottom edges
    N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 2, 0, 0, 2
    AV_FRAC = 0.10

    # we place a sufficient number of vehicles to ensure they confirm with the
    # total number specified above. We also use a "right_of_way" speed mode to
    # support traffic light compliance
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(SimCarFollowingController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            max_speed=V_ENTER,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
        ),
        routing_controller=(GridRouter, {}), num_vehicles=int(
            (N_LEFT + N_RIGHT) * N_COLUMNS / 2 + (
            N_BOTTOM + N_TOP) * N_ROWS / 2))
    vehicles.add(
        veh_id="followerstopper",
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=9,
        ),
        routing_controller=(GridRouter, {}), num_vehicles=int(
            (N_LEFT + N_RIGHT) * N_COLUMNS / 2 + (
            N_BOTTOM + N_TOP) * N_ROWS / 2))

    # inflows of vehicles are place on all outer edges (listed here)
    outer_edges = []
    outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
    # outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
    outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
    # outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

    # equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
    inflow = InFlows()
    for edge in outer_edges:
        inflow.add(
            veh_type="human",
            edge=edge,
            vehs_per_hour=EDGE_INFLOW * (1-AV_FRAC),
            departLane="free",
            departSpeed=V_ENTER)
        inflow.add(
            veh_type="followerstopper",
            edge=edge,
            vehs_per_hour=EDGE_INFLOW * AV_FRAC,
            departLane="free",
            departSpeed=V_ENTER)

    flow_params = dict(
        # name of the experiment
        exp_tag="grid_av_0_{}x{}_i{}_multiagent".format(N_ROWS, N_COLUMNS,
                                                        EDGE_INFLOW),

        # name of the flow environment the experiment is running on
        env_name='MultiGridAVsPOEnv',

        # name of the scenario class the experiment is running on
        scenario="SimpleGridScenario",

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            restart_instance=True,
            sim_step=1,
            render=False,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            additional_params={
                "target_velocity": 50,
                "switch_time": 3,
                "num_observed": 2,
                "discrete": False,
                "tl_type": "actuated",
                "max_accel": 3,
                "max_decel": 3,
                "add_rl_if_exit": True,
                "reset_inflow": True,
                "inflow_base": EDGE_INFLOW,
                "inflow_delta": 200,
                "fraction_av": AV_FRAC,
                "speed_enter": V_ENTER,
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # scenario's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            no_internal_links=False,
            additional_params={
                "speed_limit": V_ENTER + 5,  # inherited from grid0 benchmark
                "grid_array": {
                    "short_length": SHORT_LENGTH,
                    "inner_length": INNER_LENGTH,
                    "long_length": LONG_LENGTH,
                    "row_num": N_ROWS,
                    "col_num": N_COLUMNS,
                    "cars_left": N_LEFT,
                    "cars_right": N_RIGHT,
                    "cars_top": N_TOP,
                    "cars_bot": N_BOTTOM,
                },
                "horizontal_lanes": 1,
                "vertical_lanes": 1,
                "traffic_lights": False,
            },
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization
        # or reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing='custom',
            shuffle=True,
        ),
    )

    upload_dir = args.upload_dir
    ALGO = args.algo

    if ALGO == 'PPO':
        alg_run, env_name, config = setup_exps_PPO(flow_params, N_ROLLOUTS,
                                                   N_CPUS, HORIZON)

    if RUN_MODE == 'local':
        ray.init(num_cpus=N_CPUS + 1)
        N_ITER = 1
        CHECKPOINT = 1
    elif RUN_MODE == 'cluster':
        ray.init(redis_address="localhost:6379")
        N_ITER = 2000
        CHECKPOINT = 25

    exp_tag = {
        'run': alg_run,
        'env': env_name,
        'checkpoint_freq': CHECKPOINT,
        "max_failures": 10,
        'stop': {
            'training_iteration': N_ITER
        },
        'config': config,
        "num_samples": 1,
    }

    if upload_dir:
        exp_tag["upload_dir"] = "s3://{}".format(upload_dir)

    run_experiments(
        {
            flow_params["exp_tag"]: exp_tag
        },
    )
