"""
(blank)
"""

import json

import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import get_registry, register_env
from ray.tune.result import DEFAULT_RESULTS_DIR as RESULTS_DIR

from flow.core.util import rllib_logger_creator
from flow.utils.rllib import make_create_env, FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.rlcontroller import RLController

# time horizon of a single rollout
HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parrallel workers
PARALLEL_ROLLOUTS = 1

# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = Vehicles()
vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {"noise": 0.2}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=21)
vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_the_ring",

    # name of the flow environment the experiment is running on
    env_name="WaveAttenuationPOEnv",

    # name of the scenario class the experiment is running on
    scenario="LoopScenario",

    # name of the generator used to create/modify network configuration files
    generator="CircleGenerator",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=0.1,
        sumo_binary="sumo-gui",
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "blank": [220, 270],
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "length": 260,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40,
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


if __name__ == "__main__":
    ray.init(redis_address="172.31.92.24:6379", redirect_output=False)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = PARALLEL_ROLLOUTS
    config["timesteps_per_batch"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [16, 16]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = HORIZON

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    logger_creator = rllib_logger_creator(RESULTS_DIR, env_name, UnifiedLogger)

    alg = ppo.PPOAgent(env=env_name, registry=get_registry(),
                       config=config, logger_creator=logger_creator)

    # Logging out flow_params to ray's experiment result folder
    json_out_file = alg.logdir + '/flow_params.json'
    with open(json_out_file, 'w') as outfile:
        json.dump(flow_params, outfile,
                  cls=FlowParamsEncoder, sort_keys=True, indent=4)

    trials = run_experiments({
        "ring_stabilize": {
            "run": "PPO",
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "max_failures": 999,
            "stop": {"training_iteration": 200},
            "repeat": 3,
            "trial_resources": {"cpu": 1, "gpu": 0,
                                "extra_cpu": PARALLEL_ROLLOUTS - 1}
        },
    })
