"""
Run script for multiagent bottleneck.
"""
import gym
import json
import os
import numpy as np

from flow.core.params import NetParams, EnvParams, InitialConfig, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams, SumoParams
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights
from flow.core.util import register_env, NameEncoder

from flow.scenarios.bridge_toll.gen import BBTollGenerator
from flow.scenarios.bridge_toll.scenario import BBTollScenario
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SumoLaneChangeController
from ray.tune import run_experiments
import ray
import ray.rllib.ppo as ppo

from ray.tune.registry import register_env as register_rllib_env

RL_VEHICLES = 8
HORIZON = 150
SCALING = 1
NUM_LANES = 4 * SCALING
DISABLE_TB = True
DISABLE_RAMP_METER = True
FLOW_RATE = 1500

vehicle_params = [dict(veh_id="rl",
                       acceleration_controller=(RLController, {}),
                       lane_change_controller=(SumoLaneChangeController, {}),
                       routing_controller=(ContinuousRouter, {}),
                       speed_mode=0b11111,
                       lane_change_mode=1621,
                       num_vehicles=int(RL_VEHICLES / 2 * SCALING),
                       sumo_car_following_params=SumoCarFollowingParams(
                           minGap=2.5, tau=1.0),
                       sumo_lc_params=SumoLaneChangeParams()),
                  dict(veh_id="human",
                       speed_mode=0b11111,
                       lane_change_controller=(SumoLaneChangeController, {}),
                       routing_controller=(ContinuousRouter, {}),
                       lane_change_mode=512,
                       sumo_car_following_params=SumoCarFollowingParams(
                           minGap=2.5, tau=1.0),
                       num_vehicles=15 * SCALING),
                  dict(veh_id="rl2",
                       acceleration_controller=(RLController, {}),
                       lane_change_controller=(SumoLaneChangeController, {}),
                       routing_controller=(ContinuousRouter, {}),
                       speed_mode=0b11111,
                       lane_change_mode=1621,
                       num_vehicles=int(RL_VEHICLES / 2 * SCALING),
                       sumo_car_following_params=SumoCarFollowingParams(
                           minGap=2.5, tau=1.0),
                       sumo_lc_params=SumoLaneChangeParams()),
                  dict(veh_id="human2",
                       speed_mode=0b11111,
                       lane_change_mode=512,
                       lane_change_controller=(SumoLaneChangeController, {}),
                       routing_controller=(ContinuousRouter, {}),
                       sumo_car_following_params=SumoCarFollowingParams(
                           minGap=2.5, tau=1.0),
                       num_vehicles=15 * SCALING)
                  ]

flow_dist = np.ones(NUM_LANES) / NUM_LANES

inflow = InFlows()
for i in range(NUM_LANES):
    lane_num = str(i)
    veh_per_hour = FLOW_RATE * flow_dist[i]
    inflow.add(veh_type="human", edge="1", vehsPerHour=veh_per_hour,
               departLane=lane_num, departSpeed=10)

traffic_lights = TrafficLights()
if not DISABLE_TB:
    traffic_lights.add(node_id="2")
if not DISABLE_RAMP_METER:
    traffic_lights.add(node_id="3")

additional_net_params = {"scaling": SCALING}
net_params = NetParams(in_flows=inflow,
                       no_internal_links=False,
                       additional_params=additional_net_params)

flow_params = dict(
    sumo=dict(
        sumo_binary="sumo", sim_step=0.5
    ),
    env=dict(vehicle_arrangement_shuffle=False,
             lane_change_duration=1,
             additional_params={"target_velocity": 50,
                                "disable_tb": DISABLE_TB,
                                "disable_ramp_metering": DISABLE_RAMP_METER,
                                "add_rl_if_exit": True}
             ),
    net=dict(
        in_flows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params
    ),
    veh=vehicle_params,
    initial=dict(spacing="uniform",
                 min_gap=5,
                 lanes_distribution=float("inf"),
                 edges_distribution=["2", "3", "4", "5"]
                 )
)


def make_create_env(flow_env_name, flow_params, version=0,
                    exp_tag="example", sumo="sumo"):
    env_name = flow_env_name + '-v%s' % version

    sumo_params_dict = flow_params['sumo']
    sumo_params = SumoParams(**sumo_params_dict)

    env_params_dict = flow_params['env']
    env_params = EnvParams(**env_params_dict)

    net_params_dict = flow_params['net']
    net_params = NetParams(**net_params_dict)

    vehicle_params = flow_params['veh']

    init_params = flow_params['initial']

    def create_env(env_config):
        vehicles = Vehicles()
        for i in range(len(vehicle_params)):
            vehicles.add(**vehicle_params[i])

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


if __name__ == "__main__":
    config = ppo.DEFAULT_CONFIG.copy()
    horizon = HORIZON
    num_cpus = 2
    n_rollouts = int(np.floor(20000 / HORIZON))
    num_iters = 1000
    flow_env_name = "m_BottleNeckEnv"
    exp_tag = "multiagent_bottleneck"  # experiment prefix

    ray.init(num_cpus=num_cpus, redirect_output=False)
    # ray.init(redis_address="172.31.27.111:6379", redirect_output=True)

    config["num_workers"] = num_cpus
    config["timesteps_per_batch"] = horizon * n_rollouts
    config["min_steps_per_task"] = 100
    config["gamma"] = 0.999  # discount rate
    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.01
    config["num_sgd_iter"] = 30
    config["sgd_stepsize"] = 5e-5
    config["clip_param"] = 0.2
    config["lambda"] = 0.1
    config["horizon"] = horizon
    config["observation_filter"] = "NoFilter"
    config["use_gae"] = True

    flow_params['flowenv'] = flow_env_name
    flow_params['exp_tag'] = exp_tag
    # filename without '.py'
    flow_params['module'] = os.path.basename(__file__)[:-3]

    config["model"].update({"fcnet_hiddens": [256, 256]})
    options = {"multiagent_obs_shapes": [4 * RL_VEHICLES +
                                         4 * NUM_LANES * SCALING
                                         + 2 * 5] * RL_VEHICLES,
               "multiagent_act_shapes": [2] * RL_VEHICLES,
               "is_shared_model": True,
               "multiagent_shared_model": True,
               "multiagent_hiddens": [[64, 64]] * RL_VEHICLES,
               'flowenv': flow_env_name,
               }
    config["model"].update({"custom_options": options})

    create_env, env_name = make_create_env(flow_env_name,
                                           flow_params,
                                           version=0,
                                           )

    # Register as rllib env
    register_rllib_env(flow_env_name + '-v0', create_env)

    # alg = ppo.PPOAgent(env=env_name, registry=get_registry(), config=config)
    # Logging out flow_params to ray's experiment result folder
    json_out_file = os.path.dirname(os.path.realpath(__file__)) + \
        '/flow_params.json'

    with open(json_out_file, 'w') as outfile:
        json.dump(flow_params, outfile, cls=NameEncoder,
                  sort_keys=True, indent=4)

    trials = run_experiments({
        "m_bottleneck": {
            "run": "PPO",
            "env": flow_env_name + '-v0',
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "max_failures": 999,
            "stop": {"training_iteration": num_iters},
            "local_dir": "/Users/eugenevinitsky/",
            "resources": {"cpu": num_cpus, "gpu": 0}
        },
    })
    json_out_file = trials[0].logdir + '/flow_params.json'
    with open(json_out_file, 'w') as outfile:
        json.dump(flow_params, outfile, cls=NameEncoder,
                  sort_keys=True, indent=4)
