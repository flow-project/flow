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
from flow.core.params import VehicleParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.rewards import miles_per_gallon, miles_per_megajoule

import flow.config as config
from flow.envs import TestEnv

# Instantiate which conditions we want to be true about the network

WANT_GHOST_CELL = True
# WANT_DOWNSTREAM_BOUNDARY = True
ON_RAMP = False
PENETRATION_RATE = 0.0
V_DES = 5.0
HORIZON = 1000
WARMUP_STEPS = 600

inflow_rate = 2050
inflow_speed = 25.5

accel_data = (IDMController, {'a': 1.3, 'b': 2.0, 'noise': 0.3})

if WANT_GHOST_CELL:
    from flow.networks.i210_subnetwork_ghost_cell import I210SubNetworkGhostCell, EDGES_DISTRIBUTION

    highway_start_edge = 'ghost0'
else:
    from flow.networks.i210_subnetwork import I210SubNetwork, EDGES_DISTRIBUTION

    highway_start_edge = "119257914"

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
            acceleration_controller=(FollowerStopper, {"v_des": V_DES,
                                                       "no_control_edges": ["ghost0", "119257908#3"]
                                                       }),
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
            acceleration_controller=(FollowerStopper, {"v_des": V_DES,
                                                       "no_control_edges": ["ghost0", "119257908#3"]
                                                       }),
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
    exp_tag='I-210_subnetwork',

    # name of the flow environment the experiment is running on
    env_name=TestEnv,

    # name of the network class the experiment is running on
    network=network,

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
        additional_params={"on_ramp": ON_RAMP, "ghost_edge": WANT_GHOST_CELL}
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

# =========================================================================== #
# Specify custom callable that is logged during simulation runtime.           #
# =========================================================================== #

edge_id = "119257908#1-AddedOnRampEdge"

def valid_ids(env, veh_ids):
    return [veh_id for veh_id in veh_ids if env.k.vehicle.get_edge(veh_id) not in ["ghost0", "119257908#3"]]

custom_callables = {
    "avg_merge_speed": lambda env: np.nan_to_num(np.mean(
        env.k.vehicle.get_speed(valid_ids(env, env.k.vehicle.get_ids())))),
    "avg_outflow": lambda env: np.nan_to_num(
        env.k.vehicle.get_outflow_rate(120)),
    # # we multiply by 5 to account for the vehicle length and by 1000 to convert
    # # into veh/km
    # "avg_density": lambda env: 5 * 1000 * len(env.k.vehicle.get_ids_by_edge(
    #     edge_id)) / (env.k.network.edge_length(edge_id)
    #                  * env.k.network.num_lanes(edge_id)),
    "mpg": lambda env: miles_per_gallon(env,  valid_ids(env, env.k.vehicle.get_ids()), gain=1.0),
    "mpj": lambda env: miles_per_megajoule(env, valid_ids(env, env.k.vehicle.get_ids()), gain=1.0),
}
