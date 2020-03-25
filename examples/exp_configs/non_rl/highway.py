"""Example of an open multi-lane network with human-driven vehicles."""

from flow.controllers import IDMController
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, SumoLaneChangeParams
from flow.core.params import VehicleParams, InFlows
from flow.envs.ring.lane_change_accel import ADDITIONAL_ENV_PARAMS
from flow.networks.highway import HighwayNetwork, ADDITIONAL_NET_PARAMS
from flow.envs import TestEnv

vehicles = VehicleParams()
vehicles.add(
        "human",
        num_vehicles=0,
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode="strategic",
        ),
        acceleration_controller=(IDMController, {
            "a": 0.3, "b": 2.0, "noise": 0.5
        }),
    )

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=10800 / 5.0,
    departLane="free",
    departSpeed=20)


flow_params = dict(
    # name of the experiment
    exp_tag='highway',

    # name of the flow environment the experiment is running on
    env_name=TestEnv,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.5
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=4000,
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        shuffle=True,
    ),
)
