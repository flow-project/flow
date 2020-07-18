"""Multi-agent highway with ramps example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.controllers import RLController
from flow.core.params import EnvParams, NetParams, InitialConfig, InFlows, \
                             VehicleParams, SumoParams, \
                             SumoCarFollowingParams, SumoLaneChangeParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks import HighwayRampsNetwork
from flow.envs.multiagent import MultiAgentHighwayPOEnv
from flow.networks.highway_ramps import ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env


# SET UP PARAMETERS FOR THE SIMULATION

# number of training iterations
N_TRAINING_ITERATIONS = 200
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of steps per rollout
HORIZON = 1500
# number of parallel workers
N_CPUS = 11

# inflow rate on the highway in vehicles per hour
HIGHWAY_INFLOW_RATE = 4000
# inflow rate on each on-ramp in vehicles per hour
ON_RAMPS_INFLOW_RATE = 450
# percentage of autonomous vehicles compared to human vehicles on highway
PENETRATION_RATE = 20


# SET UP PARAMETERS FOR THE NETWORK

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    # lengths of highway, on-ramps and off-ramps respectively
    "highway_length": 1500,
    "on_ramps_length": 250,
    "off_ramps_length": 250,
    # number of lanes on highway, on-ramps and off-ramps respectively
    "highway_lanes": 3,
    "on_ramps_lanes": 1,
    "off_ramps_lanes": 1,
    # speed limit on highway, on-ramps and off-ramps respectively
    "highway_speed": 30,
    "on_ramps_speed": 20,
    "off_ramps_speed": 20,
    # positions of the on-ramps
    "on_ramps_pos": [500],
    # positions of the off-ramps
    "off_ramps_pos": [1000],
    # probability for a vehicle to exit the highway at the next off-ramp
    "next_off_ramp_proba": 0.25
})


# SET UP PARAMETERS FOR THE ENVIRONMENT

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
additional_env_params.update({
    'max_accel': 1,
    'max_decel': 1,
    'target_velocity': 30
})


# CREATE VEHICLE TYPES AND INFLOWS

vehicles = VehicleParams()
inflows = InFlows()

# human vehicles
vehicles.add(
    veh_id="idm",
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",  # for safer behavior at the merges
        tau=1.5  # larger distance between cars
    ),
    lane_change_params=SumoLaneChangeParams(lane_change_mode=1621))

# autonomous vehicles
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}))

# add human vehicles on the highway
inflows.add(
    veh_type="idm",
    edge="highway_0",
    vehs_per_hour=HIGHWAY_INFLOW_RATE,
    depart_lane="free",
    depart_speed="max",
    name="idm_highway_inflow")

# add autonomous vehicles on the highway
# they will stay on the highway, i.e. they won't exit through the off-ramps
inflows.add(
    veh_type="rl",
    edge="highway_0",
    vehs_per_hour=int(HIGHWAY_INFLOW_RATE * PENETRATION_RATE / 100),
    depart_lane="free",
    depart_speed="max",
    name="rl_highway_inflow",
    route="routehighway_0_0")

# add human vehicles on all the on-ramps
for i in range(len(additional_net_params['on_ramps_pos'])):
    inflows.add(
        veh_type="idm",
        edge="on_ramp_{}".format(i),
        vehs_per_hour=ON_RAMPS_INFLOW_RATE,
        depart_lane="free",
        depart_speed="max",
        name="idm_on_ramp_inflow")


# SET UP FLOW PARAMETERS

flow_params = dict(
    # name of the experiment
    exp_tag='multiagent_highway',

    # name of the flow environment the experiment is running on
    env_name=MultiAgentHighwayPOEnv,

    # name of the network class the experiment is running on
    network=HighwayRampsNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=200,
        sims_per_step=1,  # do not put more than one
        additional_params=additional_env_params,
    ),

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.2,
        render=False,
        restart_instance=True
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflows,
        additional_params=additional_net_params
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


# SET UP RLLIB MULTI-AGENT FEATURES

create_env, env_name = make_create_env(params=flow_params, version=0)

# register as rllib env
register_env(env_name, create_env)

# multiagent configuration
test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


POLICY_GRAPHS = {'av': (PPOTFPolicy, obs_space, act_space, {})}

POLICIES_TO_TRAIN = ['av']


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'
