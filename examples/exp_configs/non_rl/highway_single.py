"""Example of an open network with human-driven vehicles and a wave."""

import numpy as np

from flow.controllers import IDMController
from flow.controllers.velocity_controllers import FollowerStopper
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import SumoLaneChangeParams
from flow.core.rewards import instantaneous_mpg
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
HORIZON = 1500
# whether to include noise in the car-following models
INCLUDE_NOISE = True
# penetration rate of the follower-stopper vehicles
PENETRATION_RATE = 0.0

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
    # whether to include a ghost edge. This edge is provided a different speed
    # limit.
    "use_ghost_edge": True,
    # speed limit for the ghost edge
    "ghost_speed_limit": END_SPEED,
    # length of the downstream ghost edge with the reduced speed limit
    "boundary_cell_length": 300,
})

vehicles = VehicleParams()
vehicles.add(
    "human",
    acceleration_controller=(IDMController, {
        'a': 1.3,
        'b': 2.0,
        'noise': 0.3 if INCLUDE_NOISE else 0.0,
        "fail_safe": ['obey_speed_limit', 'safe_velocity', 'feasible_accel', 'instantaneous'],
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=0.5,
        speed_mode=8
    ),
    lane_change_params=SumoLaneChangeParams(
        model="SL2015",
        lc_sublane=2.0,
    ),
)

if PENETRATION_RATE > 0.0:
    vehicles.add(
        "av",
        color='red',
        car_following_params=SumoCarFollowingParams(
            min_gap=0.5,
            speed_mode=8
        ),
        num_vehicles=0,
        acceleration_controller=(FollowerStopper, {
            "v_des": 5.0,
            "control_length": [500, 2300],
            "fail_safe": ['obey_speed_limit', 'safe_velocity', 'feasible_accel', 'instantaneous'],
        }),
    )

inflows = InFlows()

inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=int(TRAFFIC_FLOW * (1 - PENETRATION_RATE / 100)),
    depart_lane="free",
    depart_speed=TRAFFIC_SPEED,
    name="idm_highway_inflow")

if PENETRATION_RATE > 0.0:
    inflows.add(
        veh_type="av",
        edge="highway_0",
        vehs_per_hour=int(TRAFFIC_FLOW * (PENETRATION_RATE / 100)),
        depart_lane="free",
        depart_speed=TRAFFIC_SPEED,
        name="av_highway_inflow")

# SET UP FLOW PARAMETERS

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
        warmup_steps=500,
        sims_per_step=3,
    ),

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.4,
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

custom_callables = {
    "avg_merge_speed": lambda env: np.nan_to_num(np.mean(
        env.k.vehicle.get_speed(env.k.vehicle.get_ids()))),
    "avg_outflow": lambda env: np.nan_to_num(
        env.k.vehicle.get_outflow_rate(120)),
    "miles_per_gallon": lambda env: np.nan_to_num(
        instantaneous_mpg(env, env.k.vehicle.get_ids(), gain=1.0)
    )
}
