"""Example of an open multi-lane network with human-driven vehicles."""

from flow.controllers import IDMController,OV_FTL_Controller,LinearOVM,BandoFTL_Controller
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, SumoLaneChangeParams
from flow.core.params import VehicleParams, InFlows
from flow.envs.ring.lane_change_accel import ADDITIONAL_ENV_PARAMS
from flow.networks.highway import HighwayNetwork, ADDITIONAL_NET_PARAMS
from flow.envs import LaneChangeAccelEnv

accel_data = (BandoFTL_Controller,{'alpha':.5,'beta':20.0,'h_st':12.0,'h_go':50.0,'v_max':30.0,'noise':1.0})
traffic_speed = 18.1
traffic_flow = 2056

# accel_data = (IDMController,{'a':.6,'b':6.0,'noise':1.0})
# traffic_speed = 25.8
# traffic_flow = 1720

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=accel_data,
    lane_change_params=SumoLaneChangeParams(
        model="SL2015",
        lc_sublane=2.0,
    ),
)

#Does this break the sim?
vehicles.add(
    veh_id="human2",
    acceleration_controller=(LinearOVM,{'v_max':traffic_speed}),
    lane_change_params=SumoLaneChangeParams(
        model="SL2015",
        lc_sublane=2.0,
    ),
    num_vehicles=1)

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=traffic_flow,
    departLane="free",
    departSpeed=traffic_speed)

# inflow.add(
#     veh_type="human2",
#     edge="highway_0",
#     probability=0.25,
#     departLane="free",
#     departSpeed=20)


additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params['lanes'] = 1
additional_net_params['length'] = 10000




flow_params = dict(
    # name of the experiment
    exp_tag='highway',

    # name of the flow environment the experiment is running on
    env_name=LaneChangeAccelEnv,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        lateral_resolution=1.0,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=10000,
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params,
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
