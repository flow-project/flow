"""
Bottleneck in which the actions are specifying a desired velocity
in a segment of space
"""
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights

from flow.scenarios.bridge_toll.gen import BBTollGenerator
from flow.scenarios.bridge_toll.scenario import BBTollScenario
from flow.controllers.lane_change_controllers import *
from flow.controllers.velocity_controllers import FollowerStopper
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.util import NameEncoder, register_env, rllib_logger_creator

import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import get_registry, register_env as register_rllib_env
from ray.tune.result import DEFAULT_RESULTS_DIR as results_dir

import numpy as np
import json
import os
import gym

SCALING = 1
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
HORIZON = 100
AV_FRAC = 0.25

vehicle_params = [dict(veh_id="human",
                       speed_mode="all_checks",
                       lane_change_controller=(SumoLaneChangeController, {}),
                       routing_controller=(ContinuousRouter, {}),
                       lane_change_mode=512,
                       num_vehicles=5 * SCALING),
                  dict(veh_id="followerstopper",
                       acceleration_controller=(FollowerStopper,
                                                {"danger_edges": ["3", "4"]}),
                       lane_change_controller=(SumoLaneChangeController, {}),
                       routing_controller=(ContinuousRouter, {}),
                       speed_mode=9,
                       lane_change_mode=1621,
                       num_vehicles=5 * SCALING)]

num_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 1), ("5", 1)]
additional_env_params = {"target_velocity": 40,
                         "disable_tb": True, "disable_ramp_metering": True,
                         "segments": num_segments}
# flow rate
flow_rate = 4000 * SCALING
flow_dist = np.ones(NUM_LANES) / NUM_LANES

inflow = InFlows()
for i in range(NUM_LANES):
    lane_num = str(i)
    veh_per_hour = flow_rate * flow_dist[i]
    veh_per_second = veh_per_hour / 3600
    inflow.add(veh_type="human", edge="1",
               probability=veh_per_second * (1 - AV_FRAC),
               departLane=lane_num, departSpeed=23)
    inflow.add(veh_type="followerstopper", edge="1",
               probability=veh_per_second * AV_FRAC,
               departLane=lane_num, departSpeed=23)

traffic_lights = TrafficLights()
if not DISABLE_TB:
    traffic_lights.add(node_id="2")
if not DISABLE_RAMP_METER:
    traffic_lights.add(node_id="3")

additional_net_params = {"scaling": SCALING}
net_params = NetParams(in_flows=inflow,
                       no_internal_links=False, additional_params=additional_net_params)

initial_config = InitialConfig(spacing="uniform", min_gap=5,
                               lanes_distribution=float("inf"),
                               edges_distribution=["2", "3", "4", "5"])

flow_params = dict(
    sumo=dict(
        sim_step=0.5, sumo_binary="sumo-gui"
    ),
    env=dict(lane_change_duration=1, warmup_steps=80,
             sims_per_step=4, horizon=50,
             additional_params=additional_env_params
             ),
    net=dict(
        in_flows=inflow,
        no_internal_links=False, additional_params=additional_net_params
    ),
    veh=vehicle_params,
    initial=dict(
        spacing="uniform", min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"]
    ))


def make_create_env(flow_env_name, flow_params=flow_params, version=0,
                    sumo="sumo"):
    env_name = flow_env_name + '-v%s' % version

    sumo_params_dict = flow_params['sumo']
    sumo_params_dict['sumo_binary'] = sumo
    sumo_params = SumoParams(**sumo_params_dict)

    env_params_dict = flow_params['env']
    env_params = EnvParams(**env_params_dict)

    net_params_dict = flow_params['net']
    net_params = NetParams(**net_params_dict)

    init_params = flow_params['initial']

    def create_env(env_config):
        # note that the vehicles are added sequentially by the generator,
        # so place the merging vehicles after the vehicles in the ring
        vehicles = Vehicles()
        for v_param in vehicle_params:
            vehicles.add(**v_param)

        initial_config = InitialConfig(**init_params)

        scenario = BBTollScenario(name="bay_bridge_toll",
                                  generator_class=BBTollGenerator,
                                  vehicles=vehicles,
                                  net_params=net_params,
                                  initial_config=initial_config,
                                  traffic_lights=traffic_lights)

        pass_params = (flow_env_name, sumo_params, vehicles, env_params,
                       net_params, initial_config, scenario, version)

        register_env(*pass_params)
        env = gym.envs.make(env_name)

        return env

    return create_env, env_name


if __name__ == '__main__':
    config = ppo.DEFAULT_CONFIG.copy()
    horizon = HORIZON
    n_rollouts = 50

    # replace the redis address with that output by create_or_update
    # ray.init(redis_address="localhost:6379", redirect_output=False)

    parallel_rollouts = 50
    ray.init(num_cpus=parallel_rollouts, redirect_output=False)

    config["num_workers"] = parallel_rollouts  # number of parallel rollouts
    config["timesteps_per_batch"] = horizon * n_rollouts
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [16, 16]})

    config["lambda"] = 0.99
    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = horizon

    flow_env_name = "DesiredVelocityEnv"
    exp_tag = "DesiredVelocity"  # experiment prefix

    flow_params['flowenv'] = flow_env_name
    flow_params['exp_tag'] = exp_tag
    # filename without '.py'
    flow_params['module'] = os.path.basename(__file__)[:-3]

    create_env, env_name = make_create_env(flow_env_name, flow_params, version=0)

    # Register as rllib env
    register_rllib_env(env_name, create_env)

    logger_creator = rllib_logger_creator(results_dir,
                                          flow_env_name,
                                          UnifiedLogger)

    alg = ppo.PPOAgent(env=env_name, registry=get_registry(),
                       config=config, logger_creator=logger_creator)

    # Logging out flow_params to ray's experiment result folder
    json_out_file = alg.logdir + '/flow_params.json'
    with open(json_out_file, 'w') as outfile:
        json.dump(flow_params, outfile, cls=NameEncoder, sort_keys=True, indent=4)

    trials = run_experiments({
        "DesiredVelocity": {
            "run": "PPO",
            "env": "DesiredVelocityEnv-v0",
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "max_failures": 999,
            "stop": {"training_iteration": 300},
            "trial_resources": {"cpu": 1, "gpu": 0,
                                "extra_cpu": parallel_rollouts-1}
        }
    })
    json_out_file = trials[0].logdir + '/flow_params.json'
    with open(json_out_file, 'w') as outfile:
        json.dump(flow_params, outfile, cls=NameEncoder,
                  sort_keys=True, indent=4)
