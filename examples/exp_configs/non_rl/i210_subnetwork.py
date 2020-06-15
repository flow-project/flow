"""I-210 subnetwork example."""
import os
import numpy as np

from flow.controllers import IDMController
from flow.controllers import I210Router
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
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
ON_RAMP = False
# the inflow rate of vehicles (in veh/hr)
INFLOW_RATE = 2050
# the speed of inflowing vehicles from the main edge (in m/s)
INFLOW_SPEED = 25.5
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
        lane_change_mode="strategic",
    ),
    acceleration_controller=(IDMController, {
        "a": 1.3,
        "b": 2.0,
        "noise": 0.3,
    }),
    routing_controller=(I210Router, {}) if ON_RAMP else None,
)

inflow = InFlows()
# main highway
for lane in [0, 1, 2, 3, 4]:
    inflow.add(
        veh_type="human",
        edge="ghost0" if WANT_GHOST_CELL else "119257914",
        vehs_per_hour=INFLOW_RATE,
        departLane=lane,
        departSpeed=INFLOW_SPEED)
# on ramp
if ON_RAMP:
    inflow.add(
        veh_type="human",
        edge="27414345",
        vehs_per_hour=500,
        departLane="random",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="27414342#0",
        vehs_per_hour=500,
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
        color_by_speed=True,
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
custom_callables = {
    "avg_merge_speed": lambda env: np.nan_to_num(np.mean(
        env.k.vehicle.get_speed(env.k.vehicle.get_ids_by_edge(edge_id)))),
    "avg_outflow": lambda env: np.nan_to_num(
        env.k.vehicle.get_outflow_rate(120)),
    # we multiply by 5 to account for the vehicle length and by 1000 to convert
    # into veh/km
    "avg_density": lambda env: 5 * 1000 * len(env.k.vehicle.get_ids_by_edge(
        edge_id)) / (env.k.network.edge_length(edge_id)
                     * env.k.network.num_lanes(edge_id)),
}
