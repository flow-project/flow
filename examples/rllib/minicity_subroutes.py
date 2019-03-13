"""Example of modified minicity network with human-driven vehicles."""
import json
import random

import numpy as np
from matplotlib import pyplot as plt

import ray
from flow.controllers import BaseRouter, IDMController, RLController

from flow.core.params import (EnvParams, InFlows, InitialConfig, NetParams,
                              SumoCarFollowingParams, SumoLaneChangeParams,
                              SumoParams)
from flow.core.traffic_lights import TrafficLights
from flow.core.vehicles import Vehicles

from flow.scenarios.minicity import ADDITIONAL_NET_PARAMS, MiniCityScenario
from flow.scenarios.subnetworks import (SUBNET_CROP, SUBNET_IDM,
                                        SUBNET_INFLOWS, SUBNET_RL,
                                        SUBROUTE_EDGES, SubRoute)
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

# time horizon of a single rollout
HORIZON = 50
# number of rollouts per training iteration
N_ROLLOUTS = 2
# number of parallel workers
N_CPUS = 2


np.random.seed(204)

#################################################################
# CUSTOM ARCHITECTURE'drgb'
#################################################################
from ray.rllib.models import ModelCatalog, Model
import tensorflow as tf

class MyModelClass(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].

        When using dict or tuple observation spaces, you can access
        the nested sub-observation batches here as well:

        Examples:
            # >>> print(input_dict)
            {'prev_actions': <tf.Tensor shape=(?,) dtype=int64>,
             'prev_rewards': <tf.Tensor shape=(?,) dtype=float32>,
             'is_training': <tf.Tensor shape=(), dtype=bool>,
             'obs': OrderedDict([
                ('sensors', OrderedDict([
                    ('front_cam', [
                        <tf.Tensor shape=(?, 10, 10, 3) dtype=float32>,
                        <tf.Tensor shape=(?, 10, 10, 3) dtype=float32>]),
                    ('position', <tf.Tensor shape=(?, 3) dtype=float32>),
                    ('velocity', <tf.Tensor shape=(?, 3) dtype=float32>)]))])}
        """
        inputs = input_dict['obs']
        print(inputs)
        # Convolutional layer 1
        conv1 = tf.layers.conv2d(
          inputs=inputs,
          filters=16,
          kernel_size=[4, 4],
          padding="same",
          activation=tf.nn.relu)
        # Pooling layer 1
        pool1 = tf.layers.max_pooling2d(
          inputs=conv1,
          pool_size=[2, 2],
          strides=2)
        # Convolutional layer 2
        conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=16,
          kernel_size=[4, 4],
          padding="same",
          activation=tf.nn.relu)
        # Pooling layer 2
        pool2 = tf.layers.max_pooling2d(
          inputs=conv2,
          pool_size=[2, 2],
          strides=2)
        # Fully connected layer 1
        flat = tf.contrib.layers.flatten(pool2)
        fc1 = tf.layers.dense(
          inputs=flat,
          units=32,
          activation=tf.nn.sigmoid)
        # Fully connected layer 2
        fc2 = tf.layers.dense(
          inputs=fc1,
          units=num_outputs,
          activation=None)
        return fc2, fc1

    def value_function(self):
        """Builds the value function output.

        This method can be overridden to customize the implementation of the
        value function (e.g., not sharing hidden layers).

        Returns:
            Tensor of size [BATCH_SIZE] for the value function.
        """
        return tf.reshape(tf.linear(self.last_layer, 1, "value", tf.normc_initializer(1.0)), [-1])

    def custom_loss(self, policy_loss, loss_inputs):
        """Override to customize the loss function used to optimize this model.

        This can be used to incorporate self-supervised losses (by defining
        a loss over existing input and output tensors of this model), and
        supervised losses (by defining losses over a variable-sharing copy of
        this model's layers).

        You can find an runnable example in examples/custom_loss.py.

        Arguments:
            policy_loss (Tensor): scalar policy loss from the policy graph.
            loss_inputs (dict): map of input placeholders for rollout data.

        Returns:
            Scalar tensor for the customized loss for this model.
        """
        return policy_loss

    def custom_stats(self):
        """Override to return custom metrics from your model.

        The stats will be reported as part of the learner stats, i.e.,
            info:
                learner:
                    model:
                        key1: metric1
                        key2: metric2

        Returns:
            Dict of string keys to scalar tensors.
        """
        return {}

ModelCatalog.register_custom_model("my_model", MyModelClass)

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
        self.counter = 0 # Number of time steps that vehicle has not moved
        super().__init__(veh_id, router_params)

    def choose_route(self, env):
        """See parent class."""
        next_edge = None
        # modified from env.k.vehicle
        edge = env.vehicles.get_edge(self.veh_id)
        # if edge[0] == 'e_63':
        #     return ['e_63', 'e_94', 'e_52']

        subnetwork_edges = SUBROUTE_EDGES[SUBNETWORK.value]

        if edge not in subnetwork_edges:
            next_edge = None
        elif edge == self.prev_edge and self.counter < 5:
            next_edge = None
            self.counter += 1
        elif edge == self.prev_edge and self.counter >= 5:
            if type(subnetwork_edges[edge]) == str:
                next_edge = subnetwork_edges[edge]
            else:
                next_edge = random.choice(subnetwork_edges[edge])
            self.counter = 0
        elif type(subnetwork_edges[edge]) == str:
            next_edge = subnetwork_edges[edge]
            self.counter = 0
        elif type(subnetwork_edges[edge]) == list:
            next_edge = random.choice(subnetwork_edges[edge])
            self.counter = 0
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



vehicles = Vehicles()  # modified from VehicleParams

vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    routing_controller=(MinicityRouter, {}),
    sumo_car_following_params=SumoCarFollowingParams(
        decel = 4.5,
    ),
    # lane_change_params=SumoLaneChangeParams(
    #     lane_change_mode="strategic",
    # ),
    speed_mode="all_checks",
    lane_change_mode="strategic",
    initial_speed=0,
    num_vehicles=0)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(MinicityRouter, {}),
    sumo_car_following_params=SumoCarFollowingParams(
        decel = 4.5,
    ),
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
                   vehs_per_hour=100,  # Change this to modify bandwidth/traffic
                   departLane="free",
                   departSpeed=7)
        inflow.add(veh_type="rl",
                   edge=edge,
                   vehs_per_hour=1,  # Change this to modify bandwidth/traffic
                   departLane="free",
                   departSpeed=7)
    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False, additional_params=additional_net_params)
else:
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

print(SUBROUTE_EDGES[SUBNETWORK.value].keys())
initial_config = InitialConfig(
    spacing="random",
    min_gap=5,
    edges_distribution=list(SUBROUTE_EDGES[SUBNETWORK.value].keys())
)

xmin = int(input("What is xmin? 0}"))
xmax = int(input("What is xmax? 1200"))
ymin = int(input("What is ymin? 0"))
ymax = int(input("What is ymax? 1200"))
additional_env_params = {
    'target_velocity': 11,
    'switch_time': 7,
    'num_observed': 2,
    'discrete': True,
    'tl_type': 'controlled',
    'subnetwork': SUBNETWORK.value,
    'xmin':xmin,
    'xmax':xmax,
    'ymin':ymin,
    'ymax':ymax,

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
        render=RENDERER,
        restart_instance=True
        pxpm = 1
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

# TODO: look into running multiple simulations per step


def setup_exps():
    alg_run = 'DQN'

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config["timesteps_per_iteration"] = HORIZON * N_ROLLOUTS
    config["hiddens"] = [256]
    config["lr"] = 5e-4  # TODO: hp tune
    config["grad_norm_clipping"] = 40  # TODO: hp tune
    config["schedule_max_timesteps"] = 100000  # TODO: maybe try 5e5, 1e6
    config["buffer_size"] = 50000  # TODO: maybe try 1e5, 5e5
    config["target_network_update_freq"] = 500  # TODO: this is too small
    config['model']['custom_model'] = "my_model"
    config['model']['custom_options'] = {}

    # config['num_workers'] = N_CPUS
    # config['train_batch_size'] = HORIZON * N_ROLLOUTS
    # config['sample_batch_size'] = 2
    # config['sgd_minibatch_size'] = 10
    # config['gamma'] = 0.999  # discount rate
    # config['model'].update({'fcnet_hiddens': [32, 32]})
    # config['use_gae'] = True
    # config['lambda'] = 0.97
    # config['kl_target'] = 0.02
    # config['num_sgd_iter'] = 10
    # config['horizon'] = HORIZON
    # config['model']['custom_model'] = "my_model"
    # config['model']['custom_options'] = {}

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
    try:
        ray.init(num_cpus=N_CPUS + 1)
    except DeprecationWarning:
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
