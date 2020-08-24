"""Example of an open network with human-driven vehicles."""
from flow.controllers import IDMController,ACC_Switched_Controller_Attacked,IDMController_Set_Congestion
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

import numpy as np

# the speed of vehicles entering the network
TRAFFIC_SPEED = 12.0
# the maximum speed at the downstream boundary edge
END_SPEED = 10
# the inflow rate of vehicles
TRAFFIC_FLOW = 2500
# the simulation time horizon (in steps)
HORIZON = 5000
# whether to include noise in the car-following models
INCLUDE_NOISE = True
# Wht percentage of the flow should be ACC vehicles
ATTACK_PENETRATION_RATE = 0.1
# Number of warmup steps to allow congestion to build:
CONGESTION_PERIOD = 2000
# Length of the road:
ROAD_LENGTH = 2500


HORIZON += CONGESTION_PERIOD



additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    # length of the highway
    "length": ROAD_LENGTH,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # number of edges to divide the highway into
    "num_edges": 1,
    # whether to include a ghost edge. This edge is provided a different speed
    # limit.
    "use_ghost_edge": True,
    # speed limit for the ghost edge
    "ghost_speed_limit": END_SPEED,
    # length of the downstream ghost edge with the reduced speed limit
    "boundary_cell_length": 100,
})



def IDM_Equilibrium(v0=30,T=1,delta=4,s0=2,v_eq=10):
    return np.sqrt((s0 + v_eq*T)/(1 - (v_eq/v0)**delta))

# initial_num_vehicles = int(np.floor(ROAD_LENGTH/(IDM_Equilibrium(v_eq=END_SPEED)+4.0)))

# initial_num_vehicles = int(np.max([0,initial_num_vehicles-20]))

initial_num_vehicles = 149

vehicles = VehicleParams()
inflows = InFlows()
#Human Drivers:
vehicles.add(
    "human",
    acceleration_controller=(IDMController_Set_Congestion, {
        'a': 1.3,
        'b': 2.0,
        'switch_param_time': CONGESTION_PERIOD,
        'noise': 0.3 if INCLUDE_NOISE else 0.0
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=0.1
    ),
    lane_change_params=SumoLaneChangeParams(
        model="SL2015",
        lc_sublane=2.0,
    ),
    num_vehicles=initial_num_vehicles,
    initial_speed = END_SPEED,
)
inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=TRAFFIC_FLOW*(1-ATTACK_PENETRATION_RATE),
    depart_lane="free",
    depart_speed=TRAFFIC_SPEED,
    name="idm_highway_inflow")
#ACC Drivers:
if(ATTACK_PENETRATION_RATE > 0.0):
    vehicles.add("ACC",
        acceleration_controller=(ACC_Switched_Controller_Attacked, {
            'switch_param_time': CONGESTION_PERIOD,
            'noise': 0.3 if INCLUDE_NOISE else 0.0
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.1
        ),
        lane_change_params=SumoLaneChangeParams(
            model="SL2015",
            lc_sublane=2.0,
        ),
    )
    inflows.add(
        veh_type="ACC",
        edge="highway_0",
        vehs_per_hour=TRAFFIC_FLOW*(ATTACK_PENETRATION_RATE),
        depart_lane="free",
        depart_speed=TRAFFIC_SPEED,
        name="ACC_Attacked")

#Add human drivers:


#Add ACC drivers:

flow_params = dict(
    # name of the experiment
    exp_tag='highway-single',

    # name of the flow environment the experiment is running on
    env_name=TestEnv,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
    ),

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        use_ballistic=True,
        restart_instance=False
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
