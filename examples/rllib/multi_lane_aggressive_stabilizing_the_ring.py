"""Multilane, aggressive changing, 5 lane stabilization

This example consists of 80 IDM cars on a 5-lane ring creating shockwaves.
"""

import json

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.vehicles import Vehicles
from flow.controllers import RLController

from flow.controllers import SumoCarFollowingController, SafeAggressiveLaneChanger, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams, SumoCarFollowingParams, SumoLaneChangeParams

# time horizon of a single rollout
HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 8

# We place 8 autonomous vehicle and 72 human-driven vehicles in the network
vehicles = Vehicles()
vehicles.add(
    veh_id="krauss_fast",
    acceleration_controller=(SumoCarFollowingController, {}),
    sumo_car_following_params=SumoCarFollowingParams(car_follow_model="Krauss", speedDev=0.7),
    lane_change_controller=(SafeAggressiveLaneChanger, {"target_velocity": 22.25, "threshold": 0.8}),
    sumo_lc_params=SumoLaneChangeParams(lcKeepRight=0, lcAssertive=0.5,
                                        lcSpeedGain=1.5, lcSpeedGainRight=1.0),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=72)

vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=8)

flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_the_ring",

    # name of the flow environment the experiment is running on
    env_name="MultiRLWaveAttenuationPOEnv",

    # name of the scenario class the experiment is running on
    scenario="LoopScenario",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=0.1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=50,
        starting_position_shuffle=True,
        vehicle_arrangement_shuffle=True,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [400, 400],
            "num_lanes": 4
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "length": 400,
            "lanes": 4,
            "speed_limit": 30,
            "resolution": 40,
        }, ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(bunching=40, spacing="uniform"),
)

if __name__ == "__main__":
    ray.init(num_cpus=N_CPUS + 1, redirect_output=True)

    # The algorithm or model to train. This may refer to "
    #      "the name of a built-on algorithm (e.g. RLLib's DQN "
    #      "or PPO), or a user-defined trainable function or "
    #      "class registered in the tune registry.")
    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = N_CPUS
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [16, 16]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": alg_run,
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "max_failures": 999,
            "stop": {
                "training_iteration": 500,
            },
            "upload_dir": "multi-ring-stabilize"
        },
    })
