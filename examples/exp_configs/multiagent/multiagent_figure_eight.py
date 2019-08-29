"""Example of a multi-agent environment containing a figure eight.

This example consists of one autonomous vehicle and an adversary that is
allowed to perturb the accelerations of figure eight.
"""

# WARNING: Expected total reward is zero as adversary reward is
# the negative of the AV reward

from copy import deepcopy
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph

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
    exp_tag='multiagent_figure_eight',

    # name of the flow environment the experiment is running on
    env_name='MultiAgentAccelEnv',

    # name of the network class the experiment is running on
    network='Figure8Network',

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


obs_space = 2 * (N_HUMANS + N_AVS)
act_space = N_AVS


def gen_policy():
    """Generate a policy in RLlib."""
    return PPOPolicyGraph, obs_space, act_space, {}


# Setup PG with an ensemble of `num_policies` different policy graphs
policy_graphs = {'av': gen_policy(), 'adversary': gen_policy()}


def policy_mapping_fn(agent_id):
    """Map a policy in RLlib."""
    return agent_id
