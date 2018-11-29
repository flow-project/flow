"""
Example of a multi-lane network with human-driven vehicles.
"""
from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows
from flow.core.traffic_lights import TrafficLights
from flow.core.vehicles import Vehicles
from flow.controllers import ContinuousRouter, \
    SumoLaneChangeController


import json

import ray
import ray.rllib.agents.ppo as ppo

import logging
import numpy as np

# number of parallel workers
N_CPUS = 14
# number of rollouts per training iteration
N_ROLLOUTS = N_CPUS * 4
SCALING = 1 
NUM_LANES = 4*SCALING  # number of lanes in the widest highway
HORIZON=1000

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(sim_step = 0.5, sumo_binary="sumo",restart_instance=True)

vehicles = Vehicles()

vehicles.add(
    veh_id="human",
    speed_mode=9,
    lane_change_controller=(SumoLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    lane_change_mode=0,  # 1621,#0b100000101,
    num_vehicles=1 * SCALING)

additional_env_params = {"num_observed": 16,
                         "disable_ramp_metering": False,
                         "disable_tb": True,
                         "reset_inflow": False,
                         "lane_change_duration": 5,
                         "max_accel": 3,
                         "max_decel": 3,
                         "inflow_range": [1000, 2000],
                         "target_velocity": 40}

flow_rate = 1500 * SCALING
print('flow rate is ', flow_rate)

inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="1",
    vehs_per_hour=flow_rate,
    departLane="random",
    departSpeed=10)

traffic_lights = TrafficLights()
traffic_lights.add(node_id="3")

additional_net_params = {"scaling": SCALING}

flow_params = dict(
    # name of the experiment
    exp_tag="RLRampMetering",

    # name of the flow environment the experiment is running on
    env_name="RLRampMeterEnv",

    # name of the scenario class the experiment is running on
    scenario="BottleneckScenario",

    # name of the generator used to create/modify network configuration files
    generator="BottleneckGenerator",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=0.5,
        render=False,
        print_warnings=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        warmup_steps=40,
        sims_per_step=2,
        horizon=HORIZON,
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
        spacing="uniform",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"],
    ),

    # traffic lights to be introduced to specific nodes (see
    # flow.core.traffic_lights.TrafficLights)
    tls=traffic_lights,
)

if __name__ == '__main__':
    ray.init(num_cpus=N_CPUS+1, redirect_output=True)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = N_CPUS  # number of parallel rollouts
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [300, 300, 300]})
    config["lambda"] = 0.99
    config["sgd_minibatch_size"] = 64
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json

    create_env, env_name = make_create_env(flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": "PPO",
            "env": "RLRampMeterEnv-v0",
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "max_failures": 999,
            "stop": {
                "training_iteration": 400,
            },
            "num_samples": 1
        }
    })

