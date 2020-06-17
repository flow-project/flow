"""Bay Bridge simulation."""

import os
import urllib.request

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoCarFollowingParams, SumoLaneChangeParams, InFlows
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.networks.bay_bridge import EDGES_DISTRIBUTION
from flow.controllers import SimCarFollowingController, BayBridgeRouter
from flow.envs import BayBridgeEnv
from flow.networks import BayBridgeNetwork

USE_TRAFFIC_LIGHTS = False
USE_INFLOWS = False


TEMPLATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bay_bridge.net.xml")

# download the template from AWS
if USE_TRAFFIC_LIGHTS:
    my_url = "http://s3-us-west-1.amazonaws.com/flow.netfiles/" \
             "bay_bridge_TL_all_green.net.xml"
else:
    my_url = "http://s3-us-west-1.amazonaws.com/flow.netfiles/" \
             "bay_bridge_junction_fix.net.xml"
my_file = urllib.request.urlopen(my_url)
data_to_write = my_file.read()

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), TEMPLATE),
          "wb+") as f:
    f.write(data_to_write)


vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    routing_controller=(BayBridgeRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speedDev=0.2,
        speed_mode="all_checks",
    ),
    lane_change_params=SumoLaneChangeParams(
        lc_assertive=20,
        lc_pushy=0.8,
        lc_speed_gain=4.0,
        model="LC2013",
        lane_change_mode="no_lc_safe",
        # lcKeepRight=0.8
    ),
    num_vehicles=1400)

inflow = InFlows()

if USE_INFLOWS:
    # south
    inflow.add(
        veh_type="human",
        edge="183343422",
        vehsPerHour=528,
        departLane="0",
        departSpeed=20)
    inflow.add(
        veh_type="human",
        edge="183343422",
        vehsPerHour=864,
        departLane="1",
        departSpeed=20)
    inflow.add(
        veh_type="human",
        edge="183343422",
        vehsPerHour=600,
        departLane="2",
        departSpeed=20)

    inflow.add(
        veh_type="human",
        edge="393649534",
        probability=0.1,
        departLane="0",
        departSpeed=20)  # no data for this

    # west
    inflow.add(
        veh_type="human",
        edge="11189946",
        vehsPerHour=1752,
        departLane="0",
        departSpeed=20)
    inflow.add(
        veh_type="human",
        edge="11189946",
        vehsPerHour=2136,
        departLane="1",
        departSpeed=20)
    inflow.add(
        veh_type="human",
        edge="11189946",
        vehsPerHour=576,
        departLane="2",
        departSpeed=20)

    # north
    inflow.add(
        veh_type="human",
        edge="28413687#0",
        vehsPerHour=2880,
        departLane="0",
        departSpeed=20)
    inflow.add(
        veh_type="human",
        edge="28413687#0",
        vehsPerHour=2328,
        departLane="1",
        departSpeed=20)
    inflow.add(
        veh_type="human",
        edge="28413687#0",
        vehsPerHour=3060,
        departLane="2",
        departSpeed=20)
    inflow.add(
        veh_type="human",
        edge="11198593",
        probability=0.1,
        departLane="0",
        departSpeed=20)  # no data for this
    inflow.add(
        veh_type="human",
        edge="11197889",
        probability=0.1,
        departLane="0",
        departSpeed=20)  # no data for this

    # midway through bridge
    inflow.add(
        veh_type="human",
        edge="35536683",
        probability=0.1,
        departLane="0",
        departSpeed=20)  # no data for this


flow_params = dict(
    # name of the experiment
    exp_tag='bay_bridge',

    # name of the flow environment the experiment is running on
    env_name=BayBridgeEnv,

    # name of the network class the experiment is running on
    network=BayBridgeNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.6,
        overtake_right=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=1500,
        additional_params={},
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        template=TEMPLATE,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        min_gap=15,
        edges_distribution=EDGES_DISTRIBUTION.copy(),
    ),

    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    tls=TrafficLightParams(),
)
