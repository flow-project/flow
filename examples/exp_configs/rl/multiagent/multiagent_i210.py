"""Multi-agent I-210 example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
import os
import numpy as np

from ray.tune.registry import register_env

from flow.controllers import RLController
from flow.controllers.routing_controllers import I210Router
from flow.controllers.car_following_models import IDMController
import flow.config as config
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import SumoLaneChangeParams
from flow.core.rewards import energy_consumption
from flow.envs.multiagent.i210 import I210MultiEnv, ADDITIONAL_ENV_PARAMS
from flow.utils.registry import make_create_env
from flow.networks.i210_subnetwork import I210SubNetwork, EDGES_DISTRIBUTION

# =========================================================================== #
# Specify some configurable constants.                                        #
# =========================================================================== #

# whether to include the downstream slow-down edge in the network as well as a ghost cell at the upstream edge
WANT_BOUNDARY_CONDITIONS = True
# whether to include vehicles on the on-ramp
ON_RAMP = False
# the inflow rate of vehicles (in veh/hr)
INFLOW_RATE = 2050
# the inflow rate on the on-ramp (in veh/hr)
ON_RAMP_INFLOW_RATE = 500
# the speed of inflowing vehicles from the main edge (in m/s)
INFLOW_SPEED = 25.5
# fraction of vehicles that are RL vehicles. 0.10 corresponds to 10%
PENETRATION_RATE = 0.10
# desired speed of the vehicles in the network
V_DES = 5.0
# horizon over which to run the env
HORIZON = 1500
# steps to run before follower-stopper is allowed to take control
WARMUP_STEPS = 600
# whether to turn off the fail safes for the human-driven vehicles
ALLOW_COLLISIONS = False

# =========================================================================== #
# Specify the path to the network template.                                   #
# =========================================================================== #

if WANT_BOUNDARY_CONDITIONS:
    NET_TEMPLATE = os.path.join(
        config.PROJECT_PATH,
        "examples/exp_configs/templates/sumo/i210_with_ghost_cell_with_"
        "downstream.xml")
else:
    NET_TEMPLATE = os.path.join(
        config.PROJECT_PATH,
        "examples/exp_configs/templates/sumo/test2.net.xml")
edges_distribution = EDGES_DISTRIBUTION.copy()

# =========================================================================== #
# Set up parameters for the environment.                                      #
# =========================================================================== #

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
additional_env_params.update({
    'max_accel': 2.6,
    'max_decel': 4.5,

    # configure the observation space. Look at the I210MultiEnv class for more
    # info.
    'lead_obs': True,
    # whether to add in a reward for the speed of nearby vehicles
    "local_reward": True,
    # whether to use the MPG reward. Otherwise, defaults to a target velocity
    # reward
    "mpg_reward": False,
    # whether to use the MPJ reward. Otherwise, defaults to a target velocity
    # reward
    "mpj_reward": False,
    # how many vehicles to look back for any reward
    "look_back_length": 3,
    # whether to reroute vehicles once they have exited
    "reroute_on_exit": False,
    'target_velocity': 5.0,
    # how many AVs there can be at once (this is only for centralized critics)
    "max_num_agents": 10,
    # which edges we shouldn't apply control on
    "no_control_edges": ["ghost0", "119257908#3"],

    # whether to add a slight reward for opening up a gap that will be annealed
    # out N iterations in
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
    "speed_reward_gain": 0.5,
    # penalize stopped vehicles
    "penalize_stops": False,
    "stop_penalty": 0.1,

    # penalize accels
    "penalize_accel": False,
    "accel_penalty": 0.1
})

# =========================================================================== #
# Specify vehicle-specific information and inflows.                           #
# =========================================================================== #

# create the base vehicle types that will be used for inflows
vehicles = VehicleParams()
if ON_RAMP:
    vehicles.add(
        "human",
        num_vehicles=0,
        routing_controller=(I210Router, {}),
        acceleration_controller=(IDMController, {
            'a': 1.3,
            'b': 2.0,
            'noise': 0.3
        }),
        car_following_params=SumoCarFollowingParams(
            speed_mode=19 if ALLOW_COLLISIONS else 'right_of_way'
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode="strategic",
        ),
    )
else:
    vehicles.add(
        "human",
        num_vehicles=0,
        acceleration_controller=(IDMController, {
            'a': 1.3,
            'b': 2.0,
            'noise': 0.3
        }),
        car_following_params=SumoCarFollowingParams(
            speed_mode=19 if ALLOW_COLLISIONS else 'right_of_way'
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode="strategic",
        ),
    )
vehicles.add(
    "av",
    num_vehicles=0,
    acceleration_controller=(RLController, {}),
)

inflow = InFlows()
for lane in [0, 1, 2, 3, 4]:
    if WANT_BOUNDARY_CONDITIONS:
        # Add the inflows from the main highway.
        inflow.add(
            veh_type="human",
            edge="ghost0",
            vehs_per_hour=int(INFLOW_RATE * (1 - PENETRATION_RATE)),
            departLane=lane,
            departSpeed=INFLOW_SPEED)
        inflow.add(
            veh_type="av",
            edge="ghost0",
            vehs_per_hour=int(INFLOW_RATE * PENETRATION_RATE),
            departLane=lane,
            departSpeed=INFLOW_SPEED)
    else:
        # Add the inflows from the main highway.
        inflow.add(
            veh_type="human",
            edge="119257914",
            vehs_per_hour=int(INFLOW_RATE * (1 - PENETRATION_RATE)),
            departLane=lane,
            departSpeed=INFLOW_SPEED)
        inflow.add(
            veh_type="av",
            edge="119257914",
            vehs_per_hour=int(INFLOW_RATE * PENETRATION_RATE),
            departLane=lane,
            departSpeed=INFLOW_SPEED)

    # Add the inflows from the on-ramps.
    if ON_RAMP:
        inflow.add(
            veh_type="human",
            edge="27414345",
            vehs_per_hour=int(ON_RAMP_INFLOW_RATE * (1 - PENETRATION_RATE)),
            departLane="random",
            departSpeed=10)
        inflow.add(
            veh_type="human",
            edge="27414342#0",
            vehs_per_hour=int(ON_RAMP_INFLOW_RATE * (1 - PENETRATION_RATE)),
            departLane="random",
            departSpeed=10)

# =========================================================================== #
# Generate the flow_params dict with all relevant simulation information.     #
# =========================================================================== #

flow_params = dict(
    # name of the experiment
    exp_tag='I_210_subnetwork',

    # name of the flow environment the experiment is running on
    env_name=I210MultiEnv,

    # name of the network class the experiment is running on
    network=I210SubNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # simulation-related parameters
    sim=SumoParams(
        sim_step=0.4,
        render=False,
        color_by_speed=False,
        restart_instance=True,
        use_ballistic=True,
        disable_collisions=True
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=3,
        warmup_steps=WARMUP_STEPS,
        additional_params=additional_env_params,
        done_at_exit=not additional_env_params["reroute_on_exit"]
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        template=NET_TEMPLATE,
        additional_params={
            "on_ramp": ON_RAMP,
            "ghost_edge": WANT_BOUNDARY_CONDITIONS
        }
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        edges_distribution=edges_distribution,
    ),
)

# =========================================================================== #
# Set up rllib multi-agent features.                                          #
# =========================================================================== #

create_env, env_name = make_create_env(params=flow_params, version=0)

# register as rllib env
register_env(env_name, create_env)

# multi-agent configuration
test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space

POLICY_GRAPHS = {'av': (None, obs_space, act_space, {})}

POLICIES_TO_TRAIN = ['av']


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


custom_callables = {
    "avg_speed": lambda env: np.mean([
        speed for speed in
        env.k.vehicle.get_speed(env.k.vehicle.get_ids()) if speed >= 0]),
    "avg_outflow": lambda env: np.nan_to_num(
        env.k.vehicle.get_outflow_rate(120)),
    "avg_energy": lambda env: -1 * energy_consumption(env, 0.1),
    "avg_per_step_energy": lambda env: -1 * energy_consumption(
        env, 0.1) / env.k.vehicle.num_vehicles,
}
