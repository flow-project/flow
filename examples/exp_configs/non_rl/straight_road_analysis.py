"""Example of an open multi-lane network with human-driven vehicles."""

from flow.controllers import IDMController,OV_FTL,LinearOVM
from flow.controllers import StaticLaneChanger
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, SumoLaneChangeParams
from flow.core.params import VehicleParams, InFlows
from flow.envs.ring.lane_change_accel import ADDITIONAL_ENV_PARAMS
from flow.networks.highway import HighwayNetwork, ADDITIONAL_NET_PARAMS
from flow.envs import LaneChangeAccelEnv






traffic_speed = 28.0
traffic_flow = 2240

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    lane_change_controller=(StaticLaneChanger,{}),
    acceleration_controller=(OV_FTL, {'alpha':.5,'beta':20.0,'s0':12.0,'s_star':2.0,'vM':30.0,'nosie':.5}),
    )

# vehicles.add(
#     veh_id="downstream_boundary",
#     acceleration_controller=(LinearOVM,{'v_max':traffic_speed}),
#     initial_speed=traffic_speed,
#     num_vehicles=1
#     )


env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=traffic_flow,
    departSpeed=traffic_speed,
    departLane="free")


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
        sim_step=0.1,
        render=True,
        lateral_resolution=1.0,
        color_by_speed=True,
        use_ballistic=True
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
