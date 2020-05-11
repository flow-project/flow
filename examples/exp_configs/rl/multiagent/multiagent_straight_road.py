"""Multi-agent highway with ramps example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
from flow.controllers import BandoFTLController, RLController
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import SumoLaneChangeParams
from flow.networks import HighwayNetwork
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.envs.multiagent import MultiStraightRoad
from flow.networks.highway import ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env

# SET UP PARAMETERS FOR THE SIMULATION
# percentage of autonomous vehicles compared to human vehicles on highway
PENETRATION_RATE = 10

TRAFFIC_SPEED = 11
END_SPEED = 16
TRAFFIC_FLOW = 2056
HORIZON = 2000
INCLUDE_NOISE = False


# SET UP PARAMETERS FOR THE NETWORK

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    # length of the highway
    "length": 2000,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # number of edges to divide the highway into
    "num_edges": 2,
    # whether to include a ghost edge of length 500m. This edge is provided a
    # different speed limit.
    "use_ghost_edge": True,
    # speed limit for the ghost edge
    "ghost_speed_limit": END_SPEED
})


# SET UP PARAMETERS FOR THE ENVIRONMENT

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
additional_env_params.update({
    'max_accel': 2.6,
    'max_decel': 4.5,
    'target_velocity': 12.0,
    'local_reward': True,
    'lead_obs': True,
    # whether to reroute vehicles once they have exited
    "reroute_on_exit": False,
    # whether to use the MPG reward. Otherwise, defaults to a target velocity reward
    "mpg_reward": True,
    # how many AVs there can be at once (this is only for centralized critics)
    "max_num_agents": 10,

    # whether to add a slight reward for opening up a gap that will be annealed out N iterations in
    "headway_curriculum": False,
    # how many timesteps to anneal the headway curriculum over
    "headway_curriculum_iters": 100,
    # weight of the headway reward
    "headway_reward_gain": 1.0,

    # whether to add a slight reward for traveling at a desired speed
    "speed_curriculum": True,
    # how many timesteps to anneal the headway curriculum over
    "speed_curriculum_iters": 100,
    # weight of the headway reward
    "speed_reward_gain": 1.0
})


# CREATE VEHICLE TYPES AND INFLOWS

vehicles = VehicleParams()
inflows = InFlows()
vehicles.add(
    "human",
    num_vehicles=0,
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="strategic",
    ),
    acceleration_controller=(BandoFTLController, {
        'alpha': .5,
        'beta': 20.0,
        'h_st': 12.0,
        'h_go': 50.0,
        'v_max': 30.0,
        'noise': 1.0 if INCLUDE_NOISE else 0.0,
    }),
)

# autonomous vehicles
vehicles.add(
    color='red',
    veh_id='rl',
    acceleration_controller=(RLController, {}))

# add human vehicles on the highway
inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=int(TRAFFIC_FLOW * (1 - PENETRATION_RATE / 100)),
    depart_lane="free",
    depart_speed="23.0",
    name="idm_highway_inflow")

# add autonomous vehicles on the highway
# they will stay on the highway, i.e. they won't exit through the off-ramps
inflows.add(
    veh_type="rl",
    edge="highway_0",
    vehs_per_hour=int(TRAFFIC_FLOW * (PENETRATION_RATE / 100)),
    depart_lane="free",
    depart_speed="23.0",
    name="rl_highway_inflow")

# SET UP FLOW PARAMETERS
warmup_steps = 0
if additional_env_params['reroute_on_exit']:
    warmup_steps = 400

flow_params = dict(
    # name of the experiment
    exp_tag='multiagent_highway',

    # name of the flow environment the experiment is running on
    env_name=MultiStraightRoad,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=warmup_steps,
        sims_per_step=1,  # do not put more than one
        additional_params=additional_env_params,
    ),

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.5,
        render=False,
        use_ballistic=True,
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


POLICY_GRAPHS = {'av': (None, obs_space, act_space, {})}

POLICIES_TO_TRAIN = ['av']


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'
