"""I-210 subnetwork example."""
import os
import numpy as np

from flow.controllers import IDMController
from flow.controllers import I210Router
from flow.controllers.velocity_controllers import FollowerStopper
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.rewards import miles_per_gallon
import flow.config as config
from flow.envs import TestEnv
from flow.networks.i210_subnetwork import I210SubNetwork, EDGES_DISTRIBUTION

# =========================================================================== #
# Specify some configurable constants.                                        #
# =========================================================================== #

# whether to include the upstream ghost edge in the network
WANT_GHOST_CELL = True
# whether to include the downstream slow-down edge in the network
WANT_DOWNSTREAM_BOUNDARY = True
# whether to include vehicles on the on-ramp
ON_RAMP = True
# the inflow rate of vehicles (in veh/hr)
INFLOW_RATE = 5 * 2215
# the speed of inflowing vehicles from the main edge (in m/s)
INFLOW_SPEED = 24.1
# fraction of AVs
PENETRATION_RATE = 10.0

# =========================================================================== #
# Specify the path to the network template.                                   #
# =========================================================================== #

if WANT_DOWNSTREAM_BOUNDARY:
    net_template = os.path.join(
        config.PROJECT_PATH,
        "examples/exp_configs/templates/sumo/i210_with_ghost_cell_with_"
        "downstream.xml")
elif WANT_GHOST_CELL:
    net_template = os.path.join(
        config.PROJECT_PATH,
        "examples/exp_configs/templates/sumo/i210_with_ghost_cell.xml")
else:
    net_template = os.path.join(
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
# human vehicles
vehicles.add(
    "human",
    num_vehicles=0,
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="strategic",
    ),
    acceleration_controller=(IDMController, {
        "a": 1.3,
        "b": 2.0,
        "noise": 0.3,
    }),
    routing_controller=(I210Router, {}) if ON_RAMP else None,
)

if PENETRATION_RATE > 0.0:
    vehicles.add(
        "av",
        num_vehicles=0,
        acceleration_controller=(FollowerStopper, {"v_des": 12.0}),
    )


inflow = InFlows()

# main highway
inflow.add(
    veh_type="human",
    edge="ghost0" if WANT_GHOST_CELL else "119257914",
    vehs_per_hour=int(INFLOW_RATE * (1 - PENETRATION_RATE / 100)),
    departLane="best",
    departSpeed=INFLOW_SPEED)
if PENETRATION_RATE > 0.0:
    inflow.add(
        veh_type="av",
        edge="119257914",
        vehs_per_hour=int(INFLOW_RATE * (PENETRATION_RATE / 100)),
        depart_lane="free",
        depart_speed="23",
        name="av_highway_inflow")
# on ramp
if ON_RAMP:
    inflow.add(
        veh_type="human",
        edge="27414345",
        vehs_per_hour=int(500 * (1 - PENETRATION_RATE / 100)),
        departLane="random",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="27414342#0",
        vehs_per_hour=int(500 * (1 - PENETRATION_RATE / 100)),
        departLane="random",
        departSpeed=10)
    if PENETRATION_RATE > 0.0:
        inflow.add(
            veh_type="human",
            edge="av",
            vehs_per_hour=int(500 * PENETRATION_RATE / 100),
            departLane="random",
            departSpeed=10)
        inflow.add(
            veh_type="av",
            edge="27414342#0",
            vehs_per_hour=int(500 * PENETRATION_RATE / 100),
            departLane="random",
            departSpeed=10)

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
        horizon=10000,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        template=net_template,
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
custom_callables = {
    "avg_merge_speed": lambda env: np.nan_to_num(np.mean(
        env.k.vehicle.get_speed(env.k.vehicle.get_ids()))),
    "avg_outflow": lambda env: np.nan_to_num(
        env.k.vehicle.get_outflow_rate(120)),
    # we multiply by 5 to account for the vehicle length and by 1000 to convert
    # into veh/km
    "avg_density": lambda env: 5 * 1000 * len(env.k.vehicle.get_ids_by_edge(
        edge_id)) / (env.k.network.edge_length(edge_id)
                     * env.k.network.num_lanes(edge_id)),
    "mpg": lambda env: miles_per_gallon(env, env.k.vehicle.get_ids(), gain=1.0)
}
