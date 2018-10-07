"""Figure eight example."""

import json

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.scenarios.figure8.figure8_scenario import ADDITIONAL_NET_PARAMS

# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 2

# We place one autonomous vehicle and 13 human-driven vehicles in the network
vehicles = Vehicles()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    sumo_car_following_params=SumoCarFollowingParams(
        speed_mode="no_collide",
    ),
    num_vehicles=13)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    sumo_car_following_params=SumoCarFollowingParams(
        speed_mode="no_collide",
    ),
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag="figure_eight_intersection_control",

    # name of the flow environment the experiment is running on
    env_name="AccelEnv",

    # name of the scenario class the experiment is running on
    scenario="Figure8Scenario",

    # name of the generator used to create/modify network configuration files
    generator="Figure8Generator",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=0.1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": 20,
            "max_accel": 3,
            "max_decel": 3,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params=ADDITIONAL_NET_PARAMS,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)

if __name__ == "__main__":
    ray.init(num_cpus=N_CPUS+1, redirect_output=False)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = N_CPUS
    config["timesteps_per_batch"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [100, 50, 25]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = HORIZON
    config["observation_filter"] = "NoFilter"

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
            "checkpoint_freq": 1,
            "max_failures": 999,
            "stop": {
                "training_iteration": 200
            },
            "repeat": 3,
        },
    })
