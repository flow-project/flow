"""File demonstrating formation of congestion in bottleneck."""

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.params import InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.envs import BottleneckEnv
from flow.networks import BottleneckNetwork

SCALING = 1
DISABLE_TB = True

# If set to False, ALINEA will control the ramp meter
DISABLE_RAMP_METER = True
INFLOW = 2300
HORIZON = 1000

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=25,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=1621,
    ),
    num_vehicles=1)

inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="1",
    vehsPerHour=INFLOW,
    departLane="random",
    departSpeed=10)

traffic_lights = TrafficLightParams()
if not DISABLE_TB:
    traffic_lights.add(node_id="2")
if not DISABLE_RAMP_METER:
    traffic_lights.add(node_id="3")


flow_params = dict(
    # name of the experiment
    exp_tag='bay_bridge_toll',

    # name of the flow environment the experiment is running on
    env_name=BottleneckEnv,

    # name of the network class the experiment is running on
    network=BottleneckNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.5,
        render=False,
        overtake_right=False,
        restart_instance=False
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": 40,
            "max_accel": 1,
            "max_decel": 1,
            "lane_change_duration": 5,
            "add_rl_if_exit": False,
            "disable_tb": DISABLE_TB,
            "disable_ramp_metering": DISABLE_RAMP_METER
        }
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params={
            "scaling": SCALING,
            "speed_limit": 23
        }
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="random",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"]
    ),

    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    tls=traffic_lights,
)
