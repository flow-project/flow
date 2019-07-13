"""Example of an open multi-lane network with human-driven vehicles."""

from flow.controllers import IDMController
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.params import VehicleParams, InFlows
from flow.envs.loop.lane_changing import ADDITIONAL_ENV_PARAMS
from flow.scenarios.highway import ADDITIONAL_NET_PARAMS

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {}),
    num_vehicles=20)
vehicles.add(
    veh_id="human2",
    acceleration_controller=(IDMController, {}),
    num_vehicles=20)

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="highway_0",
    probability=0.25,
    departLane="free",
    departSpeed=20)
inflow.add(
    veh_type="human2",
    edge="highway_0",
    probability=0.25,
    departLane="free",
    departSpeed=20)


flow_params = dict(
    # name of the experiment
    exp_tag='highway',

    # name of the flow environment the experiment is running on
    env_name='LaneChangeAccelEnv',

    # name of the scenario class the experiment is running on
    scenario='HighwayScenario',

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=1500,
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
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
