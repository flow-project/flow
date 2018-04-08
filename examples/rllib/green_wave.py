"""
Grid/green wave example

Attributes
----------
additional_env_params : dict
    Extra environment params
additional_net_params : dict
    Extra network parameters
flow_params : dict
    Large dictionary of flow parameters for experiment,
    passed in to `make_create_env` and used to create
    `flow_params.json` file used by exp visualizer
HORIZON : int
    Length of rollout, in steps
vehicle_params : list of dict
    List of dictionaries specifying vehicle characteristics
    and the number of each vehicle
"""
import gym
import json
import os

import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune.registry import get_registry, register_env as register_rllib_env

from flow.core.util import register_env, NameEncoder

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows
from flow.core.params import SumoCarFollowingParams
from flow.core.vehicles import Vehicles

from flow.controllers.routing_controllers import *
from flow.controllers.car_following_models import SumoCarFollowingController

from flow.scenarios.grid.gen import SimpleGridGenerator
from flow.scenarios.grid.grid_scenario import SimpleGridScenario


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
    initial_config = dict(spacing="uniform",
                          lanes_distribution=4,
                          shuffle=True)
    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(veh_type="idm", edge=outer_edges[i], probability=0.25,
                   departLane="free", departSpeed=20)

    net_params = dict(in_flows=inflow, no_internal_links=False,
                      additional_params=additional_net_params)

    return initial_config, net_params


def get_non_flow_params(enter_speed, additional_net_params):
    initial_config = dict(spacing="custom",
                          additional_params={"enter_speed": enter_speed})
    net_params = dict(no_internal_links=False,
                      additional_params=additional_net_params)

    return initial_config, net_params


HORIZON = 200
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

grid_array = {"short_length": short_length, "inner_length": inner_length,
              "long_length": long_length, "row_num": n, "col_num": m,
              "cars_left": num_cars_left, "cars_right": num_cars_right,
              "cars_top": num_cars_top, "cars_bot": num_cars_bot,
              "rl_veh": rl_veh}

additional_env_params = {"target_velocity": 50, "num_steps": HORIZON,
                         "control-length": 150, "switch_time": 3.0}

additional_net_params = {"speed_limit": 35, "grid_array": grid_array,
                         "horizontal_lanes": 1, "vertical_lanes": 1,
                         "traffic_lights": True}

vehicle_params = [
    dict(veh_id="idm",
         acceleration_controller=(SumoCarFollowingController, {}),
         sumo_car_following_params=SumoCarFollowingParams(minGap=2.5),
         routing_controller=(GridRouter, {}),
         num_vehicles=tot_cars,
         speed_mode="all_checks"),
]

initial_config, net_params = \
    get_non_flow_params(v_enter, additional_net_params)

flow_params = dict(
    sumo=dict(
        sim_step=1
    ),
    env=dict(
        additional_params=additional_env_params,
        max_speed=v_enter,
        horizon=HORIZON,
    ),
    net=net_params,
    veh=vehicle_params,
    initial=initial_config
)


def make_create_env(flow_env_name, flow_params, version=0, exp_tag="example",
                    sumo="sumo"):
    env_name = flow_env_name + '-v%s' % version

    sumo_params_dict = flow_params['sumo']
    sumo_params_dict['sumo_binary'] = sumo
    sumo_params = SumoParams(**sumo_params_dict)

    env_params_dict = flow_params['env']
    env_params = EnvParams(**env_params_dict)

    net_params_dict = flow_params['net']
    net_params = NetParams(**net_params_dict)

    veh_params = flow_params['veh']

    init_params = flow_params['initial']

    def create_env(env_config):
        # note that the vehicles are added sequentially by the generator,
        # so place the merging vehicles after the vehicles in the ring
        vehicles = Vehicles()
        for i in range(len(vehicle_params)):
            vehicles.add(**vehicle_params[i])

        initial_config = InitialConfig(**init_params)

        scenario = SimpleGridScenario(name=exp_tag,
                                      generator_class=SimpleGridGenerator,
                                      vehicles=vehicles,
                                      net_params=net_params,
                                      initial_config=initial_config)

        pass_params = (flow_env_name, sumo_params, vehicles, env_params,
                       net_params, initial_config, scenario, version)

        register_env(*pass_params)
        env = gym.envs.make(env_name)

        return env

    return create_env, env_name


def main():
    config = ppo.DEFAULT_CONFIG.copy()
    horizon = HORIZON
    n_rollouts = 100
    parallel_rollouts = 48
    # ray.init(num_cpus=num_cpus, redirect_output=False)
    ray.init(redis_address="localhost:6379", redirect_output=True)

    config["num_workers"] = parallel_rollouts
    config["timesteps_per_batch"] = horizon * n_rollouts
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32]})

    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 30
    config["sgd_stepsize"] = 5e-5
    config["observation_filter"] = "NoFilter"
    config["use_gae"] = False
    config["clip_param"] = 0.2
    config["horizon"] = horizon

    flow_env_name = "GreenWaveEnv"
    exp_tag = "66"  # experiment prefix

    flow_params['flowenv'] = flow_env_name
    flow_params['exp_tag'] = exp_tag
    flow_params['module'] = os.path.basename(__file__)[:-3]

    create_env, env_name = make_create_env(flow_env_name, flow_params,
                                           version=0, exp_tag=exp_tag,
                                           sumo="sumo")

    # Register as rllib env
    register_rllib_env(env_name, create_env)

    alg = ppo.PPOAgent(env=env_name, registry=get_registry(), config=config)

    # Logging out flow_params to ray's experiment result folder
    json_out_file = alg.logdir + '/flow_params.json'
    with open(json_out_file, 'w') as outfile:
        json.dump(flow_params, outfile, cls=NameEncoder, sort_keys=True,
                  indent=4)

    # NOTE KATHY: THESE ARE ITERATIONS
    trials = run_experiments({
        "green_wave": {
            "run": "PPO",
            "env": "GreenWaveEnv-v0",
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "max_failures": 999,
            "stop": {"training_iteration": 200},
            "trial_resources": {"cpu": 1, "gpu": 0,
                                "extra_cpu": parallel_rollouts - 1}
        }
    })
    json_out_file = trials[0].logdir + '/flow_params.json'
    with open(json_out_file, 'w') as outfile:
        json.dump(flow_params, outfile, cls=NameEncoder,
                  sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
