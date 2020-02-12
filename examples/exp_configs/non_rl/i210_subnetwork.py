"""I-210 subnetwork example."""
import os

from flow.core.params import SumoParams, EnvParams, NetParams, SumoLaneChangeParams
from flow.core.params import VehicleParams, InitialConfig
from flow.core.params import InFlows
import flow.config as config
from flow.envs import TestEnv
from flow.networks.i210_subnetwork import I210SubNetwork, EDGES_DISTRIBUTION


# no vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    "human",
    num_vehicles=0,
    lane_change_params=SumoLaneChangeParams(lane_change_mode="strategic")
)

inflow = InFlows()
# main highway
inflow.add(
    veh_type="human",
    edge="119257914",
    vehs_per_hour=6869,
    # probability=1.0,
    departLane="random",
    departSpeed=10)
# on ramp
inflow.add(
    veh_type="human",
    edge="27414345",
    vehs_per_hour=575,
    departLane="random",
    departSpeed=10)


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
        sim_step=0.1,
        render=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=10000,
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
