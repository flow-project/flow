"""
(description)
"""

import json
import os

import gym

import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import get_registry, register_env as register_rllib_env
from ray.tune.result import DEFAULT_RESULTS_DIR as results_dir

from flow.core.util import NameEncoder, register_env, rllib_logger_creator
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.scenarios.figure8.gen import Figure8Generator
from flow.scenarios.figure8.figure8_scenario import Figure8Scenario
from flow.controllers.rlcontroller import RLController
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.vehicles import Vehicles

HORIZON = 1500

additional_env_params = {"target_velocity": 20}
additional_net_params = {"radius_ring": 30, "lanes": 1, "speed_limit": 30,
                         "resolution": 40}
vehicle_params = [
    dict(veh_id="rl",
         acceleration_controller=(RLController, {}),
         routing_controller=(ContinuousRouter, {}),
         num_vehicles=1),
    dict(veh_id="idm",
         acceleration_controller=(IDMController, {"noise": 0.2}),
         routing_controller=(ContinuousRouter, {}),
         num_vehicles=13)
]

flow_params = dict(
    sumo=dict(
        sim_step=0.1
    ),
    env=dict(
        horizon=HORIZON,
        additional_params=additional_env_params
    ),
    net=dict(
        no_internal_links=False,
        additional_params=additional_net_params
    ),
    veh=vehicle_params,
    initial=dict(
        spacing="uniform",
    )
)


def make_create_env(flow_env_name, flow_params=flow_params, version=0,
                    exp_tag="example", sumo="sumo"):
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
        import flow.envs as flow_envs

        # note that the vehicles are added sequentially by the generator,
        # so place the merging vehicles after the vehicles in the ring
        vehicles = Vehicles()
        for v_param in vehicle_params:
            vehicles.add(**v_param)

        initial_config = InitialConfig(**init_params)

        scenario = Figure8Scenario(exp_tag, Figure8Generator, vehicles,
                                   net_params, initial_config=initial_config)

        pass_params = (flow_env_name, sumo_params, vehicles, env_params,
                       net_params, initial_config, scenario, version)

        register_env(*pass_params)
        env = gym.envs.make(env_name)

        return env

    return create_env, env_name


if __name__ == "__main__":
    config = ppo.DEFAULT_CONFIG.copy()
    horizon = HORIZON
    n_rollouts = 2

    ray.init(num_cpus=1, redirect_output=True)
    #ray.init(redis_address="localhost:6379", redirect_output=False)

    parallel_rollouts = 1
    config["num_workers"] = parallel_rollouts
    config["timesteps_per_batch"] = horizon * n_rollouts
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [16, 16]})

    config["lambda"] = 0.97
    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = horizon

    flow_env_name = "AccelEnv"
    exp_tag = "figure8_example"  # experiment prefix

    flow_params['flowenv'] = flow_env_name
    flow_params['exp_tag'] = exp_tag
    flow_params['module'] = os.path.basename(__file__)[:-3]
    flow_json = json.dump(flow_params, outfile, cls=NameEncoder, sort_keys=True,
                  indent=4)

    create_env, env_name = make_create_env(flow_env_name, flow_params,
                                           version=0, exp_tag=exp_tag)

    # Register as rllib env
    register_rllib_env(env_name, create_env)

    logger_creator = rllib_logger_creator(results_dir,
                                          flow_env_name,
                                          UnifiedLogger)

    alg = ppo.PPOAgent(env=env_name, registry=get_registry(),
                       config=config, logger_creator=logger_creator)

    # # Logging out flow_params to ray's experiment result folder
    # json_out_file = alg.logdir + '/flow_params.json'
    # with open(json_out_file, 'w') as outfile:
    #     json.dump(flow_params, outfile, cls=NameEncoder, sort_keys=True,
    #               indent=4)

    trials = run_experiments({
        "figure_eight": {
            "run": "PPO",
            "env": "AccelEnv-v0",
            "config": {
                **config
            },
            "checkpoint_freq": 1,
            "max_failures": 999,
            "stop": {"training_iteration": 3},
            "trial_resources": {"cpu": 1, "gpu": 0,
                                "extra_cpu": parallel_rollouts - 1}
        }
    })

