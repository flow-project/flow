"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.controllers import BaseRouter
from flow.core.experiment import SumoExperiment  # Modified from Experiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, InFlows
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
# from flow.core.params import VehicleParams
from flow.core.vehicles import Vehicles  # Modified from VehicleParams
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
from flow.core.traffic_lights import TrafficLights
import numpy as np
import random


from flow.envs.minicity_env import MiniCityTrafficLightsEnv, ADDITIONAL_ENV_PARAMS
#from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
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

from flow.scenarios.subnetworks import *
from flow.envs.minicity_env import AccelCNNSubnetEnv

from matplotlib import pyplot as plt


# time horizon of a single rollout
HORIZON = 10
# number of rollouts per training iteration
N_ROLLOUTS = 1
# number of parallel workers
N_CPUS = 2


np.random.seed(204)

#################################################################
# MODIFIABLE PARAMETERS
#################################################################


SUBNETWORK = SubRoute.SUB2  # CHANGE THIS PARAMETER TO SELECT CURRENT SUBNETWORK

# Set it to SubRoute.ALL, SubRoute.TOP_LEFT, etc.

TRAFFIC_LIGHTS = True  # CHANGE THIS to True to add traffic lights to Minicity

RENDERER = 'drgb'  # 'drgb'        # PARAMETER.
# Set to True to use default Sumo renderer,
# Set to 'drgb' for Fangyu's renderer

USE_CNN = True  # Set to True to use Pixel-learning CNN agent


# Set to False for default vehicle speeds observation space


#################################################################
# Minicity Environment Instantiation Logic
#################################################################

class MinicityRouter(BaseRouter):
    """A router used to continuously re-route vehicles in minicity scenario.

    This class allows the vehicle to pick a random route at junctions.
    """

    def __init__(self, veh_id, router_params):
        self.prev_edge = None
        super().__init__(veh_id, router_params)

    def choose_route(self, env):
        """See parent class."""
        next_edge = None
        # modified from env.k.vehicle
        edge = env.vehicles.get_edge(self.veh_id)
        # if edge[0] == 'e_63':
        #     return ['e_63', 'e_94', 'e_52']
        subnetwork_edges = SUBROUTE_EDGES[SUBNETWORK.value]
        if edge not in subnetwork_edges or edge == self.prev_edge:
            next_edge = None
        elif type(subnetwork_edges[edge]) == str:
            next_edge = subnetwork_edges[edge]
        elif type(subnetwork_edges[edge]) == list:
            if type(subnetwork_edges[edge][0]) == str:
                next_edge = random.choice(subnetwork_edges[edge])
            else:
                # Edge choices weighted by integer.
                # Inefficient untested implementation, but doesn't rely on numpy.random.choice or Python >=3.6 random.choices
                next_edge = random.choice(
                    sum(([edge] * weight for edge, weight in subnetwork_edges), []))
        self.prev_edge = edge
        if next_edge is None:
            return None
        else:
            return [edge, next_edge]


def define_traffic_lights():
    tl_logic = TrafficLights(baseline=False)

    nodes = ["n_i1", 'n_i2', 'n_i3', "n_i4", 'n_i6', 'n_i7', 'n_i8']
    phases = [{"duration": "20", "state": "GGGGrrGGGGrr"},
              {"duration": "4", "state": "yyyGrryyGyrr"},
              {"duration": "20", "state": "GrrGGGGrrGGG"},
              {"duration": "4", "state": "GrryyyGrryyy"}]

    # top left traffic light
    phases_2 = [{"duration": "20", "state": "GGGrGG"},
                {"duration": "4", "state": "yyyryy"},
                {"duration": "10", "state": "rrGGGr"},
                {"duration": "4", "state": "rryyyr"}]

    # center traffic light
    phases_3 = [{"duration": "20", "state": "GGGGGrrrGGGGGrrr"},
                {"duration": "4", "state": "yyyyyrrryyyyyrrr"},
                {"duration": "20", "state": "GrrrGGGGGrrrGGGG"},
                {"duration": "4", "state": "yrrryyyyyrrryyyy"}]

    # bottom right traffic light
    phases_6 = [{"duration": "20", "state": "GGGGGrr"},
                {"duration": "4", "state": "yyGGGrr"},
                {"duration": "20", "state": "GrrrGGG"},
                {"duration": "4", "state": "Grrryyy"}]

    # top right traffic light
    phases_8 = [{"duration": "20", "state": "GrrrGGG"},
                {"duration": "4", "state": "Grrryyy"},
                {"duration": "20", "state": "GGGGGrr"},
                {"duration": "4", "state": "yyGGGrr"}]

    for node_id in nodes:
        if node_id == 'n_i2':
            tl_logic.add(node_id, phases=phases_2,
                         tls_type="actuated", programID=1)
        elif node_id == 'n_i3':
            tl_logic.add(node_id, phases=phases_3,
                         tls_type="actuated", programID=1)
        elif node_id == 'n_i6':
            tl_logic.add(node_id, phases=phases_6,
                         tls_type="actuated", programID=1)
        elif node_id == 'n_i8':
            tl_logic.add(node_id, phases=phases_8,
                         tls_type="actuated", programID=1)
        else:
            tl_logic.add(node_id, phases=phases,
                         tls_type="actuated", programID=1)

    return tl_logic


pxpm = 1
sim_params = SumoParams(sim_step=0.25, emission_path='./data/')


if pxpm is not None:
    sim_params.pxpm = pxpm

vehicles = Vehicles()  # modified from VehicleParams
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    routing_controller=(MinicityRouter, {}),
    # car_following_params=SumoCarFollowingParams(
    #     speed_mode=1,
    # ),
    # lane_change_params=SumoLaneChangeParams(
    #     lane_change_mode="strategic",
    # ),
    speed_mode="all_checks",
    lane_change_mode="strategic",
    initial_speed=0,
    num_vehicles=SUBNET_IDM[SUBNETWORK.value])
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(MinicityRouter, {}),
    # car_following_params=SumoCarFollowingParams(
    #     speed_mode="strategic",
    # ),
    speed_mode="all_checks",
    lane_change_mode="strategic",
    initial_speed=0,
    num_vehicles=SUBNET_RL[SUBNETWORK.value])


additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params = {
    'speed_limit': 35,
    'horizontal_lanes': 1,
    'vertical_lanes': 1,
    'traffic_lights': True
}


# Add inflows only on edges at border of subnetwork
if len(SUBNET_INFLOWS[SUBNETWORK.value]) > 0:
    inflow = InFlows()
    for edge in SUBNET_INFLOWS[SUBNETWORK.value]:
        assert edge in SUBROUTE_EDGES[SUBNETWORK.value].keys()
        inflow.add(veh_type="idm",
                   edge=edge,
                   vehs_per_hour=1000,  # Change this to modify bandwidth/traffic
                   departLane="free",
                   departSpeed=7.5)
        inflow.add(veh_type="rl",
                   edge=edge,
                   vehs_per_hour=1,  # Change this to modify bandwidth/traffic
                   departLane="free",
                   departSpeed=7.5)
    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False, additional_params=additional_net_params)
else:
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

initial_config = InitialConfig(
    spacing="random",
    min_gap=5,
    edges_distribution=list(SUBROUTE_EDGES[SUBNETWORK.value].keys())
)


additional_env_params = {
    'target_velocity': 50,
    'switch_time': 7,
    'num_observed': 2,
    'discrete': False,
    'tl_type': 'controlled',
    'subnetwork': SUBNETWORK.value
}


initial_config = InitialConfig(
    spacing="random",
    min_gap=5
)


flow_params = dict(
    # name of the experiment
    exp_tag='minicity',

    # name of the flow environment the experiment is running on
    env_name='AccelCNNSubnetTrainingEnv',

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
