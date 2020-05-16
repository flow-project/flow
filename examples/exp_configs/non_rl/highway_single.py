"""Example of an open network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import SumoLaneChangeParams
from flow.core.params import SumoCarFollowingParams
from flow.networks import HighwayNetwork
from flow.envs import TestEnv
from flow.networks.highway import ADDITIONAL_NET_PARAMS

# the speed of vehicles entering the network
TRAFFIC_SPEED = 24.1
# the maximum speed at the downstream boundary edge
END_SPEED = 6.0
# the inflow rate of vehicles
TRAFFIC_FLOW = 2215
# the simulation time horizon (in steps)
HORIZON = 4500
# whether to include noise in the car-following models
INCLUDE_NOISE = True

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
    # whether to include a ghost edge
    "use_ghost_edge": True,
    # speed limit for the ghost edge
    "ghost_speed_limit": END_SPEED,
    # length of the cell imposing a boundary
    "boundary_cell_length": 300,
})

vehicles = VehicleParams()
vehicles.add(
    "human",
    acceleration_controller=(IDMController, {
        'a': 1.3,
        'b': 2.0,
        'noise': 0.3 if INCLUDE_NOISE else 0.0
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=0.5
    ),
    lane_change_params=SumoLaneChangeParams(
        model="SL2015",
        lc_sublane=2.0,
    ),
)

inflows = InFlows()
inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=TRAFFIC_FLOW,
    departLane="free",
    departSpeed=TRAFFIC_SPEED)

flow_params = dict(
    # name of the experiment
    exp_tag='highway',

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
        sim_step=0.4,
        render=False,
        restart_instance=False,
        use_ballistic=True
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
