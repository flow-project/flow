"""Ring road example.

Trains a number of autonomous vehicles to stabilize the flow of 22 vehicles in
a variable length ring road.
"""
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.envs.multiagent import MultiAgentWaveAttenuationPOEnv
from flow.networks import RingNetwork
from flow.utils.registry import make_create_env

# time horizon of a single rollout
HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 2
# number of automated vehicles. Must be less than or equal to 22.
NUM_AUTOMATED = 2


# We evenly distribute the automated vehicles in the network.
num_human = 22 - NUM_AUTOMATED
humans_remaining = num_human

vehicles = VehicleParams()
for i in range(NUM_AUTOMATED):
    # Add one automated vehicle.
    vehicles.add(
        veh_id="rl_{}".format(i),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1)

    # Add a fraction of the remaining human vehicles.
    vehicles_to_add = round(humans_remaining / (NUM_AUTOMATED - i))
    humans_remaining -= vehicles_to_add
    vehicles.add(
        veh_id="human_{}".format(i),
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=vehicles_to_add)


flow_params = dict(
    # name of the experiment
    exp_tag="multiagent_ring",

    # name of the flow environment the experiment is running on
    env_name=MultiAgentWaveAttenuationPOEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=False
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        clip_actions=False,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [220, 270],
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "length": 260,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40,
        }, ),

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
POLICY_GRAPHS = {'av': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


POLICIES_TO_TRAIN = ['av']
