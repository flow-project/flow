"""Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
merges in an open network.

"""
import json
import math

import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import get_registry, register_env
from ray.tune.result import DEFAULT_RESULTS_DIR as RESULTS_DIR

from flow.core.util import rllib_logger_creator
from flow.utils.rllib import make_create_env, FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows
from flow.scenarios.merge.scenario import ADDITIONAL_NET_PARAMS
from flow.core.vehicles import Vehicles
from flow.controllers.car_following_models import IDMController
from flow.controllers.rlcontroller import RLController

# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parrallel workers
PARALLEL_ROLLOUTS = 10

# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = 0.1
# initial number of vehicles
NUM_VEH = 5
# initial number of AVs
NUM_AV = math.floor(RL_PENETRATION * NUM_VEH)


# We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params["merge_lanes"] = 1
additional_net_params["highway_lanes"] = 1
additional_net_params["pre_merge_length"] = 500

# RL vehicles constitute 5% of the total number of vehicles
vehicles = Vehicles()
vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {"noise": 0.2}),
             num_vehicles=NUM_VEH-NUM_AV)
vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
             num_vehicles=NUM_AV)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
inflow.add(veh_type="human", edge="inflow_highway",
           vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
           departLane="free", departSpeed=10)
inflow.add(veh_type="rl", edge="inflow_highway",
           vehs_per_hour=RL_PENETRATION * FLOW_RATE,
           departLane="free", departSpeed=10)
inflow.add(veh_type="human", edge="inflow_merge", vehs_per_hour=100,
           departLane="free", departSpeed=7.5)

flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_open_network_merges",

    # name of the flow environment the experiment is running on
    env_name="WaveAttenuationMergePOEnv",

    # name of the scenario class the experiment is running on
    scenario="MergeScenario",

    # name of the generator used to create/modify network configuration files
    generator="MergeGenerator",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=0.2,
        sumo_binary="sumo-gui",
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=5,
        warmup_steps=0,
        additional_params={
            "max_accel": 1.5,
            "max_decel": 1.5,
            "target_velocity": 20,
            "num_rl": 5,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        in_flows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


if __name__ == "__main__":
    ray.init(num_cpus=PARALLEL_ROLLOUTS, redirect_output=True)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = PARALLEL_ROLLOUTS
    config["timesteps_per_batch"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32, 32]})
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
        "highway_stabilize": {
            "run": "PPO",
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 5,
            "max_failures": 999,
            "stop": {
                "training_iteration": 200,
            },
            "repeat": 3,
            "trial_resources": {
                "cpu": 1,
                "gpu": 0,
                "extra_cpu": PARALLEL_ROLLOUTS - 1,
            },
        },
    })
