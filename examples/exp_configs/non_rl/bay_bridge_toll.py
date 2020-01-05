"""Bay Bridge toll example."""

import os
import urllib.request

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoLaneChangeParams, SumoCarFollowingParams, InFlows
from flow.core.params import VehicleParams
from flow.networks.bay_bridge_toll import EDGES_DISTRIBUTION
from flow.controllers import SimCarFollowingController, BayBridgeRouter
from flow.envs import BayBridgeEnv
from flow.networks import BayBridgeTollNetwork

USE_TRAFFIC_LIGHTS = False

TEMPLATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bottleneck.net.xml")


# download the template from AWS
if USE_TRAFFIC_LIGHTS:
    my_url = "http://s3-us-west-1.amazonaws.com/flow.netfiles/" \
             "bay_bridge_TL_all_green.net.xml"
else:
    my_url = "http://s3-us-west-1.amazonaws.com/flow.netfiles/" \
             "bay_bridge_junction_fix.net.xml"
my_file = urllib.request.urlopen(my_url)
data_to_write = my_file.read()

with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), TEMPLATE),
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
        model="LC2013",
        lcCooperative=0.2,
        lcSpeedGain=15,
        lane_change_mode="no_lat_collide",
    ),
    num_vehicles=50)


inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="393649534",
    probability=0.2,
    departLane="random",
    departSpeed=10)
inflow.add(
    veh_type="human",
    edge="4757680",
    probability=0.2,
    departLane="random",
    departSpeed=10)
inflow.add(
    veh_type="human",
    edge="32661316",
    probability=0.2,
    departLane="random",
    departSpeed=10)
inflow.add(
    veh_type="human",
    edge="90077193#0",
    vehs_per_hour=2000,
    departLane="random",
    departSpeed=10)


flow_params = dict(
    # name of the experiment
    exp_tag='bay_bridge_toll',

    # name of the flow environment the experiment is running on
    env_name=BayBridgeEnv,

    # name of the network class the experiment is running on
    network=BayBridgeTollNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.4,
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
)
