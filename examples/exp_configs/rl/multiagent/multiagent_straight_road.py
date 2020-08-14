"""Multi-agent highway with ramps example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
from flow.controllers import RLController, IDMController
from flow.core.params import EnvParams, NetParams, InitialConfig, InFlows, \
                             VehicleParams, SumoParams, SumoLaneChangeParams, SumoCarFollowingParams
from flow.networks import HighwayNetwork
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.envs.multiagent import MultiStraightRoad
from flow.networks.highway import ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env

# SET UP PARAMETERS FOR THE SIMULATION

# the speed of vehicles entering the network
TRAFFIC_SPEED = 24.1
# the maximum speed at the downstream boundary edge
END_SPEED = 6.0
# the inflow rate of vehicles
HIGHWAY_INFLOW_RATE = 2215
# the simulation time horizon (in steps)
HORIZON = 1000
# whether to include noise in the car-following models
INCLUDE_NOISE = True

PENETRATION_RATE = 10.0

WARMUP_STEPS = 500

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    # length of the highway
    "length": 2500,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # number of edges to divide the highway into
    "num_edges": 2,
    # whether to include a ghost edge
    "use_ghost_edge": True,
    # speed limit for the ghost edge
    "ghost_speed_limit": END_SPEED,
    # length of the cell imposing a boundary
    "boundary_cell_length": 300,
})


# SET UP PARAMETERS FOR THE ENVIRONMENT

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
additional_env_params.update({
    'max_accel': 2.6,
    'max_decel': 4.5,
    'target_velocity': 6.0,
    'local_reward': True,
    'lead_obs': True,
    'control_range': [500, 2300],
    # whether to reroute vehicles once they have exited
    "reroute_on_exit": False,
    # whether to use the MPG reward. Otherwise, defaults to a target velocity reward
    "mpg_reward": False,
    # whether to use the joules reward. Otherwise, defaults to a target velocity reward
    "mpj_reward": False,
    # how many vehicles to look back for the MPG reward
    "look_back_length": 3,
    # how many AVs there can be at once (this is only for centralized critics)
    "max_num_agents": 10,

    # whether to add a slight reward for opening up a gap that will be annealed out N iterations in
    "headway_curriculum": False,
    # how many timesteps to anneal the headway curriculum over
    "headway_curriculum_iters": 100,
    # weight of the headway reward
    "headway_reward_gain": 2.0,
    # desired time headway
    "min_time_headway": 2.0,

    # whether to add a slight reward for traveling at a desired speed
    "speed_curriculum": True,
    # how many timesteps to anneal the headway curriculum over
    "speed_curriculum_iters": 20,
    # weight of the headway reward
    "speed_reward_gain": 1.0,

    # penalize stopped vehicles
    "penalize_stops": True,
    "stop_penalty": 0.05,

    # penalize accels
    "penalize_accel": True,
    "accel_penalty": 0.05,

    # control the range of speeds of the downstream edge
    # if true, we vary the downstream speed sampling uniformly from max and min at the start of
    # each rollout
    "randomize_downstream_speed": True,
    "max_downstream_speed": 5.0,
    "min_downstream_speed": 5.0,
})


# CREATE VEHICLE TYPES AND INFLOWS

vehicles = VehicleParams()
inflows = InFlows()
vehicles.add(
    "human",
    acceleration_controller=(IDMController, {
        'a': 1.3,
        'b': 2.0,
        'noise': 0.3 if INCLUDE_NOISE else 0.0,
        "fail_safe": ['obey_speed_limit', 'safe_velocity', 'feasible_accel'],
    }),
    car_following_params=SumoCarFollowingParams(
        speed_mode=12,  # right of way at intersections + obey limits on deceleration
        min_gap=0.5
    ),
)

# autonomous vehicles
default_controller = (IDMController, {
        "a": 1.3,
        "b": 2.0,
        "noise": 0.3,
        "fail_safe": ['obey_speed_limit', 'safe_velocity', 'feasible_accel'],
    }
)
vehicles.add(
    color='red',
    veh_id='rl',
    car_following_params=SumoCarFollowingParams(
        min_gap=0.5,
        speed_mode=12  # right of way at intersections + obey limits on deceleration
    ),
    acceleration_controller=(RLController, {
        "fail_safe": ['obey_speed_limit', 'safe_velocity', 'feasible_accel'],
        "default_controller": default_controller
    }))

# add human vehicles on the highway
inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=int(HIGHWAY_INFLOW_RATE * (1 - PENETRATION_RATE / 100)),
    depart_lane="free",
    depart_speed=TRAFFIC_SPEED,
    name="idm_highway_inflow")

# add autonomous vehicles on the highway
# they will stay on the highway, i.e. they won't exit through the off-ramps
inflows.add(
    veh_type="rl",
    edge="highway_0",
    vehs_per_hour=int(HIGHWAY_INFLOW_RATE * (PENETRATION_RATE / 100)),
    depart_lane="free",
    depart_speed=TRAFFIC_SPEED,
    name="rl_highway_inflow")

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
        warmup_steps=WARMUP_STEPS,
        sims_per_step=3,
        additional_params=additional_env_params
    ),

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.4,
        render=False,
        restart_instance=True,
        use_ballistic=True
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
