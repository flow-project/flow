"""Grid/green wave example."""

import json

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.controllers import SumoCarFollowingController, GridRouter

# time horizon of a single rollout
HORIZON = 200
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 2


def gen_edges(row_num, col_num):
    edges = []
    for i in range(col_num):
        edges += ["left" + str(row_num) + '_' + str(i)]
        edges += ["right" + '0' + '_' + str(i)]

    # build the left and then the right edges
    for i in range(row_num):
        edges += ["bot" + str(i) + '_' + '0']
        edges += ["top" + str(i) + '_' + str(col_num)]

    return edges


def get_flow_params(col_num, row_num, additional_net_params):
    initial_config = InitialConfig(
        spacing="uniform", lanes_distribution=float("inf"), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type="idm",
            edge=outer_edges[i],
            probability=0.25,
            departLane="free",
            departSpeed=20)

    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    return initial_config, net_params


def get_non_flow_params(enter_speed, additional_net_params):
    additional_init_params = {"enter_speed": enter_speed}
    initial_config = InitialConfig(additional_params=additional_init_params)
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    return initial_config, net_params


v_enter = 30

inner_length = 800
long_length = 100
short_length = 800
n = 1
m = 5
num_cars_left = 3
num_cars_right = 3
num_cars_top = 15
num_cars_bot = 15
rl_veh = 0
tot_cars = (num_cars_left + num_cars_right) * m \
           + (num_cars_bot + num_cars_top) * n

grid_array = {
    "short_length": short_length,
    "inner_length": inner_length,
    "long_length": long_length,
    "row_num": n,
    "col_num": m,
    "cars_left": num_cars_left,
    "cars_right": num_cars_right,
    "cars_top": num_cars_top,
    "cars_bot": num_cars_bot,
    "rl_veh": rl_veh
}

additional_env_params = {
        "target_velocity": 50,
        "switch_time": 3.0,
        "num_observed": 2,
        "discrete": False,
        "tl_type": "controlled"
    }

additional_net_params = {
    "speed_limit": 35,
    "grid_array": grid_array,
    "horizontal_lanes": 1,
    "vertical_lanes": 1
}

vehicles = Vehicles()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(SumoCarFollowingController, {}),
    sumo_car_following_params=SumoCarFollowingParams(
        minGap=2.5,
        max_speed=v_enter,
        speed_mode="all_checks",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=tot_cars)

initial_config, net_params = \
    get_non_flow_params(v_enter, additional_net_params)

flow_params = dict(
    # name of the experiment
    exp_tag="green_wave",

    # name of the flow environment the experiment is running on
    env_name="PO_TrafficLightGridEnv",

    # name of the scenario class the experiment is running on
    scenario="SimpleGridScenario",

    # name of the generator used to create/modify network configuration files
    generator="SimpleGridGenerator",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=1,
        render=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=initial_config,
)

if __name__ == "__main__":
    ray.init(num_cpus=N_CPUS+1, redirect_output=True)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = N_CPUS
    config["timesteps_per_batch"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32]})
    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 30
    config["sgd_stepsize"] = 5e-5
    config["observation_filter"] = "NoFilter"
    config["use_gae"] = True
    config["clip_param"] = 0.2
    config["horizon"] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": "PPO",
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "max_failures": 999,
            "stop": {
                "training_iteration": 200,
            },

        }
    })
