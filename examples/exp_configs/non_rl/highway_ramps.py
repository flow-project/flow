"""Example of a highway section network with on/off ramps."""

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import InFlows, VehicleParams, TrafficLightParams
from flow.networks.highway_ramps import ADDITIONAL_NET_PARAMS
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks import HighwayRampsNetwork

additional_net_params = ADDITIONAL_NET_PARAMS.copy()

# lengths
additional_net_params["highway_length"] = 1200
additional_net_params["on_ramps_length"] = 200
additional_net_params["off_ramps_length"] = 200

# number of lanes
additional_net_params["highway_lanes"] = 3
additional_net_params["on_ramps_lanes"] = 1
additional_net_params["off_ramps_lanes"] = 1

# speed limits
additional_net_params["highway_speed"] = 30
additional_net_params["on_ramps_speed"] = 20
additional_net_params["off_ramps_speed"] = 20

# ramps
additional_net_params["on_ramps_pos"] = [400]
additional_net_params["off_ramps_pos"] = [800]

# probability of exiting at the next off-ramp
additional_net_params["next_off_ramp_proba"] = 0.25

# inflow rates in vehs/hour
HIGHWAY_INFLOW_RATE = 4000
ON_RAMPS_INFLOW_RATE = 350

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",  # for safer behavior at the merges
        tau=1.5  # larger distance between cars
    ),
    lane_change_params=SumoLaneChangeParams(lane_change_mode=1621)
)

inflows = InFlows()
inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=HIGHWAY_INFLOW_RATE,
    depart_lane="free",
    depart_speed="max",
    name="highway_flow")
for i in range(len(additional_net_params["on_ramps_pos"])):
    inflows.add(
        veh_type="human",
        edge="on_ramp_{}".format(i),
        vehs_per_hour=ON_RAMPS_INFLOW_RATE,
        depart_lane="first",
        depart_speed="max",
        name="on_ramp_flow")


flow_params = dict(
    # name of the experiment
    exp_tag='highway-ramp',

    # name of the flow environment the experiment is running on
    env_name=AccelEnv,

    # name of the network class the experiment is running on
    network=HighwayRampsNetwork,

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
        horizon=3600,
        sims_per_step=5,
        warmup_steps=0
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

    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    tls=TrafficLightParams(),
)
