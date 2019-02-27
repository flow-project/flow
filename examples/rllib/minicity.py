"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.envs.minicity_env import MiniCityTrafficLightsEnv, ADDITIONAL_ENV_PARAMS
#from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import MinicityRouter
from flow.core.traffic_lights import TrafficLights
import numpy as np

import json

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.controllers import SumoCarFollowingController, GridRouter

np.random.seed(204)

# time horizon of a single rollout
HORIZON = 10
# number of rollouts per training iteration
N_ROLLOUTS = 1
# number of parallel workers
N_CPUS = 2


"""
Perform a simulation of vehicles on modified minicity of University of
Delaware.

Parameters
----------
render: bool, optional
    specifies whether to use sumo's gui during execution

Returns
-------
exp: flow.core.SumoExperiment type
    A non-rl experiment demonstrating the performance of human-driven
    vehicles on the minicity scenario.
"""
sumo_params = SumoParams(render=False)


vehicles = Vehicles()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(RLController, {}),
    routing_controller=(MinicityRouter, {}),
    speed_mode="right_of_way",
    lane_change_mode="aggressive",
    initial_speed=0,
    num_vehicles=110)
# vehicles.add(
#     veh_id="rl",
#     acceleration_controller=(RLController, {}),
#     routing_controller=(MinicityRouter, {}),
#     speed_mode="right_of_way",
#     initial_speed=0,
#     num_vehicles=10)

tl_logic = TrafficLights(baseline=False)

nodes = ["n_i1", 'n_i2', 'n_i3', "n_i4", 'n_i6', 'n_i7', 'n_i8', 'n_m3']
# phases = [{"duration": "10", "state": "GGGGrrGGGGrr"},
#           {"duration": "2", "state": "yyyGrryyGyrr"},
#           {"duration": "10", "state": "GrrGGGGrrGGG"},
#           {"duration": "2", "state": "GrryyyGrryyy"}]

# #merge
# phases_m3 = [{"duration": "10", "state": "GGrG"},
#           {"duration": "2", "state": "yGry"},
#           {"duration": "5", "state": "rGGr"},
#           {"duration": "2", "state": "rGyr"}]

# #top left traffic light
# phases_2 = [{"duration": "10", "state": "GGGrGG"},
#           {"duration": "2", "state": "yyyryy"},
#           {"duration": "5", "state": "rrGGGr"},
#           {"duration": "2", "state": "rryyyr"}]

# #center traffic light
# phases_3 = [{"duration": "10", "state": "GGGGGrrrGGGGGrrr"},
#             {"duration": "2", "state": "yyyyyrrryyyyyrrr"},
#             {"duration": "10", "state": "GrrrGGGGGrrrGGGG"},
#             {"duration": "2", "state": "yrrryyyyyrrryyyy"}]

# #bottom right traffic light
# phases_6 = [{"duration": "10", "state": "GGGGGrr"},
#             {"duration": "2", "state": "yyGGGrr"},
#             {"duration": "5", "state": "GrrrGGG"},
#             {"duration": "2", "state": "Grrryyy"}]

# #top right traffic light
# phases_8 = [{"duration": "10", "state": "GrrrGGG"},
#             {"duration": "2", "state": "Grrryyy"},
#             {"duration": "5", "state": "GGGGGrr"},
#             {"duration": "2", "state": "yyGGGrr"}]

# for node_id in nodes:
#     if node_id == 'n_i2':
#         tl_logic.add(node_id, phases=phases_2,
#                      tls_type="actuated", programID=1)
#     elif node_id == 'n_i3':
#         tl_logic.add(node_id, phases=phases_3,
#                      tls_type="actuated", programID=1)
#     elif node_id == 'n_i6':
#         tl_logic.add(node_id, phases=phases_6,
#                      tls_type="actuated", programID=1)
#     elif node_id == 'n_i8':
#         tl_logic.add(node_id, phases=phases_8,
#                      tls_type="actuated", programID=1)
#     elif node_id == 'n_m3':
#         tl_logic.add(node_id, phases=phases_m3,
#                      tls_type="actuated", programID=1)
#     else:
#         tl_logic.add(node_id, phases=phases,
#                      tls_type="actuated", programID=1)
#tl_logic.add("n_i4", tls_type="actuated", programID=1)
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)


additional_net_params = ADDITIONAL_NET_PARAMS.copy()

additional_net_params = {
    'speed_limit': 35,
    'horizontal_lanes': 1,
    'vertical_lanes': 1,
    'traffic_lights': True
}

net_params = NetParams(
    no_internal_links=False, additional_params=additional_net_params)


additional_env_params = {
    'target_velocity': 50,
    'switch_time': 7,
    'num_observed': 2,
    'discrete': False,
    'tl_type': 'controlled'
}

initial_config = InitialConfig(
    spacing="random",
    min_gap=5
)


# env = MiniCityTrafficLightsEnv(env_params, sumo_params, scenario)


flow_params = dict(
    # name of the experiment
    exp_tag='minicity',

    # name of the flow environment the experiment is running on
    env_name='MiniCityTrafficLightsEnv',

    # name of the scenario class the experiment is running on
    scenario='MiniCityScenario',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=1,
        render=False,
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

    # tls=tl_logic,
)


def setup_exps():

    alg_run = 'PPO'

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['sample_batch_size'] = 2
    config['sgd_minibatch_size'] = 10
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [32, 32]})
    config['use_gae'] = True
    config['lambda'] = 0.97
    config['kl_target'] = 0.02
    config['num_sgd_iter'] = 10
    config['horizon'] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == '__main__':
    alg_run, gym_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS + 1, redirect_output=False)
    trials = run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': gym_name,
            'config': {
                **config
            },
            'checkpoint_freq': 20,
            'max_failures': 999,
            'stop': {
                'training_iteration': 200,
            },
        }
    })
