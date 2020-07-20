"""I-210 subnetwork example."""
import os
import numpy as np

from flow.controllers.car_following_models import IDMController
from flow.controllers.velocity_controllers import FollowerStopper
from flow.controllers.routing_controllers import I210Router
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import SumoLaneChangeParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.rewards import instantaneous_mpg
from flow.networks import I210SubNetwork
from flow.networks.i210_subnetwork import EDGES_DISTRIBUTION
from flow.envs import TestEnv
import flow.config as config

# =========================================================================== #
# Specify some configurable constants.                                        #
# =========================================================================== #

# whether to include the upstream ghost edge in the network
WANT_GHOST_CELL = True
# whether to include the downstream slow-down edge in the network
WANT_DOWNSTREAM_BOUNDARY = True
# whether to include vehicles on the on-ramp
ON_RAMP = False
# the inflow rate of vehicles (in veh/hr)
INFLOW_RATE = 2050
# on-ramp inflow_rate
ON_RAMP_INFLOW_RATE = 500
# the speed of inflowing vehicles from the main edge (in m/s)
INFLOW_SPEED = 25.5
# fraction of vehicles that are follower-stoppers. 0.10 corresponds to 10%
PENETRATION_RATE = 0.0
# desired speed of the follower stopper vehicles
V_DES = 5.0
# horizon over which to run the env
HORIZON = 1500
# steps to run before follower-stopper is allowed to take control
WARMUP_STEPS = 600

# =========================================================================== #
# Specify the path to the network template.                                   #
# =========================================================================== #

if WANT_DOWNSTREAM_BOUNDARY:
    NET_TEMPLATE = os.path.join(
        config.PROJECT_PATH,
        "examples/exp_configs/templates/sumo/i210_with_ghost_cell_with_"
        "downstream.xml")
elif WANT_GHOST_CELL:
    NET_TEMPLATE = os.path.join(
        config.PROJECT_PATH,
        "examples/exp_configs/templates/sumo/i210_with_ghost_cell.xml")
else:
    NET_TEMPLATE = os.path.join(
        config.PROJECT_PATH,
        "examples/exp_configs/templates/sumo/test2.net.xml")

# If the ghost cell is not being used, remove it from the initial edges that
# vehicles can be placed on.
edges_distribution = EDGES_DISTRIBUTION.copy()
if not WANT_GHOST_CELL:
    edges_distribution.remove("ghost0")

# =========================================================================== #
# Specify vehicle-specific information and inflows.                           #
# =========================================================================== #

vehicles = VehicleParams()

vehicles.add(
    "human",
    num_vehicles=0,
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="sumo_default",
    ),
    # this is only right of way on
    car_following_params=SumoCarFollowingParams(
        speed_mode=8
    ),
    acceleration_controller=(IDMController, {
        "a": 1.3,
        "b": 2.0,
        "noise": 0.3,
        "failsafe": ['obey_speed_limit', 'safe_velocity', 'feasible_accel', 'instantaneous'],
    }),
    routing_controller=(I210Router, {}) if ON_RAMP else None,
)

vehicles.add(
    "av",
    num_vehicles=0,
    color="red",
    # this is only right of way on
    car_following_params=SumoCarFollowingParams(
        speed_mode=8
    ),
    acceleration_controller=(FollowerStopper, {
        "v_des": V_DES,
        "no_control_edges": ["ghost0", "119257908#3"],
        "failsafe": ['obey_speed_limit', 'safe_velocity', 'feasible_accel', 'instantaneous'],
    }),
    routing_controller=(I210Router, {}) if ON_RAMP else None,
)

inflow = InFlows()

# main highway
highway_start_edge = "ghost0" if WANT_GHOST_CELL else "119257914"

for lane in [0, 1, 2, 3, 4]:
    inflow.add(
        veh_type="human",
        edge=highway_start_edge,
        vehs_per_hour=INFLOW_RATE * (1 - PENETRATION_RATE),
        depart_lane=lane,
        depart_speed=INFLOW_SPEED)

    if PENETRATION_RATE > 0.0:
        inflow.add(
            veh_type="av",
            edge=highway_start_edge,
            vehs_per_hour=INFLOW_RATE * PENETRATION_RATE,
            depart_lane=lane,
            depart_speed=INFLOW_SPEED)

# on ramp
if ON_RAMP:
    inflow.add(
        veh_type="human",
        edge="27414345",
        vehs_per_hour=int(ON_RAMP_INFLOW_RATE * (1 - PENETRATION_RATE)),
        depart_speed=10,
    )

    if PENETRATION_RATE > 0.0:
        inflow.add(
            veh_type="av",
            edge="27414345",
            vehs_per_hour=int(ON_RAMP_INFLOW_RATE * PENETRATION_RATE),
            depart_lane="random",
            depart_speed=10)

# =========================================================================== #
# Generate the flow_params dict with all relevant simulation information.     #
# =========================================================================== #

flow_params = dict(
    # name of the experiment
    exp_tag='I-210_subnetwork',

    # name of the flow environment the experiment is running on
    env_name=TestEnv,

    # name of the network class the experiment is running on
    network=I210SubNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # simulation-related parameters
    sim=SumoParams(
        sim_step=0.4,
        render=False,
        color_by_speed=False,
        use_ballistic=True
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=WARMUP_STEPS,
        sims_per_step=3
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        template=NET_TEMPLATE,
        additional_params={
            "on_ramp": ON_RAMP,
            "ghost_edge": WANT_GHOST_CELL,
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
# Specify custom callable that is logged during simulation runtime.           #
# =========================================================================== #

edge_id = "119257908#1-AddedOnRampEdge"


def valid_ids(env, veh_ids):
    """Return the names of vehicles within the controllable edges."""
    return [
        veh_id for veh_id in veh_ids
        if env.k.vehicle.get_edge(veh_id) not in ["ghost0", "119257908#3"]
    ]


custom_callables = {
    "avg_merge_speed": lambda env: np.nan_to_num(np.mean(
        env.k.vehicle.get_speed(valid_ids(env, env.k.vehicle.get_ids())))),
    "avg_outflow": lambda env: np.nan_to_num(
        env.k.vehicle.get_outflow_rate(120)),
    "mpg": lambda env: instantaneous_mpg(
        env,  valid_ids(env, env.k.vehicle.get_ids()), gain=1.0),
}
