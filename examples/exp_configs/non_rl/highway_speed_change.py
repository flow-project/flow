"""
Perform a simulation of vehicles on a highway with a congested downstream
condition that can be specified by the user. This done by setting the downstream
node's speed limit to be whatever is desired. Vehicles that hit the downstream node
suddenly need to break which sets off a shockwave.
"""

from flow.controllers import IDMController, CFMController
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv, \
    ADDITIONAL_ENV_PARAMS

from flow.networks.highway_speed_change import HighwaySpeedChange, ADDITIONAL_NET_PARAMS

LANES = 2
LENGTH = 5000
V_DOWNSTREAM = 20
INFLOW_VALS = [1500, 1500, 1500]

additional_net_params = ADDITIONAL_NET_PARAMS.copy()

speed_limit = 35  # The highways default speed limit with no congestion
end_speed_limit = V_DOWNSTREAM  # what the desired congestion speed is

additional_net_params["length"] = LENGTH
additional_net_params["lanes"] = LANES
additional_net_params["speed_limit"] = speed_limit
additional_net_params["end_speed_limit"] = end_speed_limit

vehicles = VehicleParams()

# Add driver specifically for down stream speed control:
vehicles.add(
    veh_id="downstream_boundary",
    acceleration_controller=(CFMController, {'v_des': V_DOWNSTREAM}),
    initial_speed=V_DOWNSTREAM, num_vehicles=LANES)

# Add inflows for other types of vehicles:
inflow = InFlows()

# General idea here is to have three classes of drivers so that some natural
# congestion might arise from driver interaction. Unclear on to what extent
# this really makes a difference.

# Normal Drivers:
lc_assertive = 1
lc_pushy = 1
lc_speed_gain = 1
lc_KeepRight = 1
lc_pushy_gap = 5

lane_change_params_normal = SumoLaneChangeParams(
    lc_assertive=lc_assertive,
    lc_pushy=lc_pushy,
    lc_speed_gain=lc_speed_gain,
    model="LC2013",
    lane_change_mode="no_lat_collide",
    lc_keep_right=lc_KeepRight,
    lc_pushy_gap=lc_pushy_gap
)

v0 = 30.0
s0 = 2.0
T = 1.8
noise = 0.3

vehicles.add(
    veh_id="human_normal",
    acceleration_controller=(IDMController, {'v0': v0, 's0': s0, 'T': T, 'noise': noise}),
    lane_change_params=lane_change_params_normal)

inflow.add(
    veh_type="human_normal",
    edge="highway_0",
    vehsPerHour=INFLOW_VALS[0],
    departLane="random",
    departSpeed=V_DOWNSTREAM)

# Conservative Drivers

if INFLOW_VALS[1] > 0:
    lc_assertive = .1
    lc_pushy = 0.01
    lc_speed_gain = 5
    lc_KeepRight = 5
    lc_pushy_gap = 10

    lane_change_params_conservative = SumoLaneChangeParams(
        lc_assertive=lc_assertive,
        lc_pushy=lc_pushy,
        lc_speed_gain=lc_speed_gain,
        model="LC2013",
        lane_change_mode="no_lat_collide",
        lc_keep_right=lc_KeepRight,
        lc_pushy_gap=lc_pushy_gap
    )

    vehicles.add(
        veh_id="human_conservative",
        acceleration_controller=(IDMController, {'v0': 27, 's0': 2, 'T': 2.5}),
        lane_change_params=lane_change_params_conservative)

    inflow.add(
        veh_type="human_conservative",
        edge="highway_0",
        vehsPerHour=INFLOW_VALS[1],
        departLane="random",
        departSpeed=V_DOWNSTREAM)

# Aggressive Drivers:

if INFLOW_VALS[2] > 0:
    lc_assertive = 5
    lc_pushy = 5
    lc_speed_gain = .5
    lc_KeepRight = .1
    lc_pushy_gap = .5

    lane_change_params_aggressive = SumoLaneChangeParams(
        lc_assertive=lc_assertive,
        lc_pushy=lc_pushy,
        lc_speed_gain=lc_speed_gain,
        model="LC2013",
        lane_change_mode="no_lat_collide",
        lc_keep_right=lc_KeepRight,
        lc_pushy_gap=lc_pushy_gap
    )

    vehicles.add(
        veh_id="human_aggressive",
        acceleration_controller=(IDMController, {'v0': 33, 's0': 2, 'T': 1.0, "noise": 0.4}),
        lane_change_params=lane_change_params_aggressive)

    inflow.add(
        veh_type="human_aggressive",
        edge="highway_0",
        vehsPerHour=INFLOW_VALS[2],
        departLane="random",
        departSpeed=V_DOWNSTREAM)

flow_params = dict(
    # name of the experiment
    exp_tag='highway-ramp',

    # name of the flow environment the experiment is running on
    env_name=LaneChangeAccelEnv,

    # name of the network class the experiment is running on
    network=HighwaySpeedChange,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        emission_path="./data/",
        sim_step=0.2,
        restart_instance=True
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        additional_params=ADDITIONAL_ENV_PARAMS,
        horizon=500
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        shuffle=True,
        edges_distribution=["highway_0"]
    ),
)
