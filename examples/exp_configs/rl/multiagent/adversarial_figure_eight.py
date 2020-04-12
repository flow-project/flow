"""Example of a multi-agent environment containing a figure eight.

This example consists of one autonomous vehicle and an adversary that is
allowed to perturb the accelerations of figure eight.
"""

# WARNING: Expected total reward is zero as adversary reward is
# the negative of the AV reward

from copy import deepcopy
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.controllers import ContinuousRouter
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs.multiagent import AdversarialAccelEnv
from flow.networks import FigureEightNetwork
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env

# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 4
# number of parallel workers
N_CPUS = 2
# number of human-driven vehicles
N_HUMANS = 13
# number of automated vehicles
N_AVS = 1

# We place one autonomous vehicle and 13 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id='human',
    acceleration_controller=(IDMController, {
        'noise': 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='obey_safe_speed',
    ),
    num_vehicles=N_HUMANS)
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='obey_safe_speed',
    ),
    num_vehicles=N_AVS)

flow_params = dict(
    # name of the experiment
    exp_tag='adversarial_figure_eight',

    # name of the flow environment the experiment is running on
    env_name=AdversarialAccelEnv,

    # name of the network class the experiment is running on
    network=FigureEightNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            'target_velocity': 20,
            'max_accel': 3,
            'max_decel': 3,
            'perturb_weight': 0.03,
            'sort_vehicles': False
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=deepcopy(ADDITIONAL_NET_PARAMS),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return PPOTFPolicy, obs_space, act_space, {}


# Setup PG with an ensemble of `num_policies` different policy graphs
POLICY_GRAPHS = {'av': gen_policy(), 'adversary': gen_policy()}


def policy_mapping_fn(agent_id):
    """Map a policy in RLlib."""
    return agent_id
