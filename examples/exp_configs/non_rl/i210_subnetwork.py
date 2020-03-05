"""I-210 subnetwork example."""
import os

import numpy as np

from flow.controllers.routing_controllers import I210Router
from flow.controllers.car_following_models import IDMController
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

# create the base vehicle type that will be used for inflows
vehicles = VehicleParams()
vehicles.add(
    "human",
    num_vehicles=0,
    routing_controller=(I210Router, {}),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="strategic",
    ),
    acceleration_controller=(IDMController, {
        "a": 0.3, "b": 2.0, "noise": 0.5
    }),
)

inflow = InFlows()
# main highway
inflow.add(
    veh_type="human",
    edge="119257914",
    vehs_per_hour=8378,
    departLane="random",
    departSpeed=10)
# on ramp
# inflow.add(
#     veh_type="human",
#     edge="27414345",
#     vehs_per_hour=321,
#     departLane="random",
#     departSpeed=20)
# inflow.add(
#     veh_type="human",
#     edge="27414342#0",
#     vehs_per_hour=421,
#     departLane="random",
#     departSpeed=20)

NET_TEMPLATE = os.path.join(
    config.PROJECT_PATH,
    "examples/exp_configs/templates/sumo/test2.net.xml")

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
        sim_step=0.8,
        render=False,
        color_by_speed=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=4500,
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
