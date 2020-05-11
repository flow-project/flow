"""Multi-agent I-210 example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
import os
import numpy as np

from ray.tune.registry import register_env

from flow.controllers import RLController
from flow.controllers.car_following_models import IDMController
import flow.config as config
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import SumoLaneChangeParams
from flow.core.rewards import energy_consumption
from flow.networks.i210_subnetwork import I210SubNetwork, EDGES_DISTRIBUTION
from flow.envs.multiagent.i210 import I210MultiEnv, ADDITIONAL_ENV_PARAMS
from flow.utils.registry import make_create_env

# SET UP PARAMETERS FOR THE SIMULATION

# number of steps per rollout
HORIZON = 2000

VEH_PER_HOUR_BASE_119257914 = 10800
VEH_PER_HOUR_BASE_27414345 = 321
VEH_PER_HOUR_BASE_27414342 = 421


# percentage of autonomous vehicles compared to human vehicles on highway
PENETRATION_RATE = 10

# SET UP PARAMETERS FOR THE ENVIRONMENT
additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
additional_env_params.update({
    'max_accel': 2.6,
    'max_decel': 4.5,
    # configure the observation space. Look at the I210MultiEnv class for more info.
    'lead_obs': True,
    # whether to add in a reward for the speed of nearby vehicles
    "local_reward": True,
    # whether to use the MPG reward. Otherwise, defaults to a target velocity reward
    "mpg_reward": True,
    # whether to reroute vehicles once they have exited
    "reroute_on_exit": True,
    'target_velocity': 12.0,
    # how many AVs there can be at once (this is only for centralized critics)
    "max_num_agents": 10,
    # whether to add a slight reward for opening up a gap that will be annealed out N iterations in
    "headway_curriculum": False,
    # how many timesteps to anneal the headway curriculum over
    "headway_curriculum_iters": 100,
    # weight of the headway reward
    "headway_reward_gain": 0.1,

    # whether to add a slight reward for traveling at a desired speed
    "speed_curriculum": True,
    # how many timesteps to anneal the headway curriculum over
    "speed_curriculum_iters": 100,
    # weight of the headway reward
    "speed_reward_gain": 0.5
})

# CREATE VEHICLE TYPES AND INFLOWS
# no vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    "human",
    num_vehicles=0,
    lane_change_params=SumoLaneChangeParams(lane_change_mode="strategic"),
    acceleration_controller=(IDMController, {"a": .3, "b": 2.0, "noise": 0.5}),
)
vehicles.add(
    "av",
    acceleration_controller=(RLController, {}),
    num_vehicles=0,
)

inflow = InFlows()
# main highway
pen_rate = PENETRATION_RATE / 100
assert pen_rate < 1.0, "your penetration rate is over 100%"
assert pen_rate > 0.0, "your penetration rate should be above zero"
inflow.add(
    veh_type="human",
    edge="119257914",
    vehs_per_hour=int(VEH_PER_HOUR_BASE_119257914 * (1 - pen_rate)),
    # probability=1.0,
    depart_lane="random",
    departSpeed=20)
# # on ramp
# inflow.add(
#     veh_type="human",
#     edge="27414345",
#     vehs_per_hour=321 * pen_rate,
#     depart_lane="random",
#     depart_speed=20)
# inflow.add(
#     veh_type="human",
#     edge="27414342#0",
#     vehs_per_hour=421 * pen_rate,
#     depart_lane="random",
#     depart_speed=20)

# Now add the AVs
# main highway
inflow.add(
    veh_type="av",
    edge="119257914",
    vehs_per_hour=int(VEH_PER_HOUR_BASE_119257914 * pen_rate),
    # probability=1.0,
    depart_lane="random",
    depart_speed=20)
# # on ramp
# inflow.add(
#     veh_type="av",
#     edge="27414345",
#     vehs_per_hour=int(VEH_PER_HOUR_BASE_27414345 * pen_rate),
#     depart_lane="random",
#     depart_speed=20)
# inflow.add(
#     veh_type="av",
#     edge="27414342#0",
#     vehs_per_hour=int(VEH_PER_HOUR_BASE_27414342 * pen_rate),
#     depart_lane="random",
#     depart_speed=20)

NET_TEMPLATE = os.path.join(
    config.PROJECT_PATH,
    "examples/exp_configs/templates/sumo/test2.net.xml")

warmup_steps = 0
if additional_env_params['reroute_on_exit']:
    warmup_steps = 1200

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
        sim_step=0.5,
        render=False,
        color_by_speed=False,
        restart_instance=True,
        use_ballistic=True
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=1,
        warmup_steps=warmup_steps,
        additional_params=additional_env_params,
        done_at_exit=False
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        template=NET_TEMPLATE
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        edges_distribution=EDGES_DISTRIBUTION,
    ),
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


custom_callables = {
    "avg_speed": lambda env: np.mean([speed for speed in
                                      env.k.vehicle.get_speed(env.k.vehicle.get_ids()) if speed >= 0]),
    "avg_outflow": lambda env: np.nan_to_num(
        env.k.vehicle.get_outflow_rate(120)),
    "avg_energy": lambda env: -1*energy_consumption(env, 0.1)
}
