"""Ring road example.

Creates a set of stabilizing the ring experiments to test if
 more agents -> fewer needed batches
"""
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.controllers import ContinuousRouter
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.params import VehicleParams
from flow.envs.multiagent import MultiWaveAttenuationPOEnv
from flow.networks import MultiRingNetwork
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env

# make sure (sample_batch_size * num_workers ~= train_batch_size)
# time horizon of a single rollout
HORIZON = 3000
# Number of rings
NUM_RINGS = 1
# number of rollouts per training iteration
N_ROLLOUTS = 20  # int(20/NUM_RINGS)
# number of parallel workers
N_CPUS = 2  # int(20/NUM_RINGS)

# We place one autonomous vehicle and 21 human-driven vehicles in the network
vehicles = VehicleParams()
for i in range(NUM_RINGS):
    vehicles.add(
        veh_id='human_{}'.format(i),
        acceleration_controller=(IDMController, {
            'noise': 0.2
        }),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=21)
    vehicles.add(
        veh_id='rl_{}'.format(i),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag='lord_of_numrings{}'.format(NUM_RINGS),

    # name of the flow environment the experiment is running on
    env_name=MultiWaveAttenuationPOEnv,

    # name of the network class the experiment is running on
    network=MultiRingNetwork,

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
        warmup_steps=750,
        additional_params={
            'max_accel': 1,
            'max_decel': 1,
            'ring_length': [230, 230],
            'target_velocity': 4
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            'length': 230,
            'lanes': 1,
            'speed_limit': 30,
            'resolution': 40,
            'num_rings': NUM_RINGS
        }, ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(bunching=20.0, spacing='custom'),
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
