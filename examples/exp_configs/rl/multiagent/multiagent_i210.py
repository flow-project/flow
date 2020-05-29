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
from flow.core.params import SumoLaneChangeParams
from flow.core.params import SumoCarFollowingParams
from flow.core.rewards import energy_consumption
from flow.networks.i210_subnetwork import I210SubNetwork, EDGES_DISTRIBUTION
from flow.envs.multiagent.i210 import I210MultiEnv, ADDITIONAL_ENV_PARAMS
from flow.utils.registry import make_create_env

# SET UP PARAMETERS FOR THE SIMULATION
WANT_GHOST_CELL = True
# WANT_DOWNSTREAM_BOUNDARY = True
ON_RAMP = False
PENETRATION_RATE = 0.10
V_DES = 5.0
HORIZON = 1000
WARMUP_STEPS = 600

inflow_rate = 2050
inflow_speed = 25.5

accel_data = (IDMController, {'a': 1.3, 'b': 2.0, 'noise': 0.3})

VEH_PER_HOUR_BASE_119257914 = 10800
VEH_PER_HOUR_BASE_27414345 = 321
VEH_PER_HOUR_BASE_27414342 = 421

if WANT_GHOST_CELL:
    from flow.networks.i210_subnetwork_ghost_cell import I210SubNetworkGhostCell, EDGES_DISTRIBUTION

    edges_distribution = EDGES_DISTRIBUTION
    highway_start_edge = 'ghost0'
else:
    from flow.networks.i210_subnetwork import I210SubNetwork, EDGES_DISTRIBUTION
    edges_distribution = EDGES_DISTRIBUTION
    highway_start_edge = "119257914"

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
    "mpg_reward": False,
    # whether to use the MPJ reward. Otherwise, defaults to a target velocity reward
    "mpj_reward": False,
    # how many vehicles to look back for any reward
    "look_back_length": 10,
    # whether to reroute vehicles once they have exited
    "reroute_on_exit": False,
    'target_velocity': 5.0,
    # how many AVs there can be at once (this is only for centralized critics)
    "max_num_agents": 10,
    # which edges we shouldn't apply control on
    "no_control_edges": ["ghost0", "119257908#3"],

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
    "speed_reward_gain": 0.5,
    # penalize stopped vehicles
    "penalize_stops": True,
    "stop_penalty": 0.05,

    # penalize accels
    "penalize_accel": True,
    "accel_penalty": 0.05
})

# CREATE VEHICLE TYPES AND INFLOWS
# no vehicles in the network
vehicles = VehicleParams()

inflow = InFlows()

if ON_RAMP:
    vehicles.add(
        "human",
        num_vehicles=0,
        color="white",
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode="strategic",
        ),
        acceleration_controller=accel_data,
        routing_controller=(I210Router, {})
    )
    if PENETRATION_RATE > 0.0:
        vehicles.add(
            "av",
            num_vehicles=0,
            color="red",
            acceleration_controller=(RLController, {}),
            routing_controller=(I210Router, {})
        )

    # inflow.add(
    #     veh_type="human",
    #     edge=highway_start_edge,
    #     vehs_per_hour=inflow_rate,
    #     departLane="best",
    #     departSpeed=inflow_speed)

    lane_list = ['0', '1', '2', '3', '4']

    for lane in lane_list:
        inflow.add(
            veh_type="human",
            edge=highway_start_edge,
            vehs_per_hour=int(inflow_rate * (1 - PENETRATION_RATE)),
            departLane=lane,
            departSpeed=inflow_speed)

    inflow.add(
        veh_type="human",
        edge="27414345",
        vehs_per_hour=int(500 * (1 - PENETRATION_RATE)),
        departLane="random",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="27414342#0",
        vehs_per_hour=int(500 * (1 - PENETRATION_RATE)),
        departLane="random",
        departSpeed=10)

    if PENETRATION_RATE > 0.0:
        for lane in lane_list:
            inflow.add(
                veh_type="av",
                edge=highway_start_edge,
                vehs_per_hour=int(inflow_rate * PENETRATION_RATE),
                departLane=lane,
                departSpeed=inflow_speed)

        inflow.add(
            veh_type="av",
            edge="27414345",
            vehs_per_hour=int(500 * PENETRATION_RATE),
            departLane="random",
            departSpeed=10)
        inflow.add(
            veh_type="av",
            edge="27414342#0",
            vehs_per_hour=int(500 * PENETRATION_RATE),
            departLane="random",
            departSpeed=10)

else:
    # create the base vehicle type that will be used for inflows
    vehicles.add(
        "human",
        num_vehicles=0,
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode="strategic",
        ),
        acceleration_controller=accel_data,
    )
    if PENETRATION_RATE > 0.0:
        vehicles.add(
            "av",
            color="red",
            num_vehicles=0,
            acceleration_controller=(RLController, {}),
        )

    # If you want to turn off the fail safes uncomment this:

    # vehicles.add(
    #     'human',
    #     num_vehicles=0,
    #     lane_change_params=SumoLaneChangeParams(
    #         lane_change_mode='strategic',
    #     ),
    #     acceleration_controller=accel_data,
    #     car_following_params=SumoCarFollowingParams(speed_mode='19')
    # )

    lane_list = ['0', '1', '2', '3', '4']

    for lane in lane_list:
        inflow.add(
            veh_type="human",
            edge=highway_start_edge,
            vehs_per_hour=int(inflow_rate * (1 - PENETRATION_RATE)),
            departLane=lane,
            departSpeed=inflow_speed)

    if PENETRATION_RATE > 0.0:
        for lane in lane_list:
            inflow.add(
                veh_type="av",
                edge=highway_start_edge,
                vehs_per_hour=int(inflow_rate * PENETRATION_RATE),
                departLane=lane,
                departSpeed=inflow_speed)


network_xml_file = "examples/exp_configs/templates/sumo/i210_with_ghost_cell_with_downstream_test.xml"

# network_xml_file = "examples/exp_configs/templates/sumo/i210_with_congestion.xml"

NET_TEMPLATE = os.path.join(config.PROJECT_PATH, network_xml_file)

if WANT_GHOST_CELL:
    network = I210SubNetworkGhostCell
else:
    network = I210SubNetwork

flow_params = dict(
    # name of the experiment
    exp_tag='I_210_subnetwork',

    # name of the flow environment the experiment is running on
    env_name=I210MultiEnv,

    # name of the network class the experiment is running on
    network=network,

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
            "ghost_edge": WANT_GHOST_CELL
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
    "avg_outflow": lambda env: np.nan_to_num(env.k.vehicle.get_outflow_rate(120)),
    "avg_energy": lambda env: -1*energy_consumption(env, 0.1),
    "avg_per_step_energy": lambda env: -1*energy_consumption(env, 0.1) / env.k.vehicle.num_vehicles,
}
