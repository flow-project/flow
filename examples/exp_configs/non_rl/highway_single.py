"""Multi-agent highway with ramps example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
import numpy as np

from flow.controllers import BandoFTLController
from flow.controllers.velocity_controllers import FollowerStopper
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import SumoLaneChangeParams
from flow.core.rewards import miles_per_megajoule
from flow.networks import HighwayNetwork
from flow.envs import TestEnv
from flow.networks.highway import ADDITIONAL_NET_PARAMS

TRAFFIC_SPEED = 11
END_SPEED = 16
TRAFFIC_FLOW = 2056
HORIZON = 2000
INCLUDE_NOISE = False

# percentage of autonomous vehicles compared to human vehicles on highway
PENETRATION_RATE = 10.0

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
    # whether to include a ghost edge of length 500m. This edge is provided a
    # different speed limit.
    "use_ghost_edge": True,
    # speed limit for the ghost edge
    "ghost_speed_limit": END_SPEED
})

vehicles = VehicleParams()
vehicles.add(
    "human",
    num_vehicles=0,
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="strategic",
    ),
    acceleration_controller=(BandoFTLController, {
        'alpha': .5,
        'beta': 20.0,
        'h_st': 12.0,
        'h_go': 50.0,
        'v_max': 30.0,
        'noise': 1.0 if INCLUDE_NOISE else 0.0,
    }),
)

if PENETRATION_RATE > 0.0:
    vehicles.add(
        "av",
        num_vehicles=0,
        acceleration_controller=(FollowerStopper, {"v_des": 11.0}),
    )

inflows = InFlows()

inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=int(TRAFFIC_FLOW * (1 - PENETRATION_RATE / 100)),
    depart_lane="free",
    depart_speed="23",
    name="idm_highway_inflow")

if PENETRATION_RATE > 0.0:
    inflows.add(
        veh_type="av",
        edge="highway_0",
        vehs_per_hour=int(TRAFFIC_FLOW * (PENETRATION_RATE / 100)),
        depart_lane="free",
        depart_speed="23",
        name="av_highway_inflow")

# SET UP FLOW PARAMETERS

flow_params = dict(
    # name of the experiment
    exp_tag='highway-single',

    # name of the flow environment the experiment is running on
    env_name=TestEnv,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=0,
        sims_per_step=1,
    ),

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.5,
        render=False,
        restart_instance=False
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

custom_callables = {
    "avg_merge_speed": lambda env: np.nan_to_num(np.mean(
        env.k.vehicle.get_speed(env.k.vehicle.get_ids()))),
    "avg_outflow": lambda env: np.nan_to_num(
        env.k.vehicle.get_outflow_rate(120)),
    "miles_per_megajoule": lambda env: np.nan_to_num(
        miles_per_megajoule(env, env.k.vehicle.get_ids(), gain=1.0)
    )
}
