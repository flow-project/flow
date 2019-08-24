"""Example of a figure 8 network with human-driven vehicles.

Right-of-way dynamics near the intersection causes vehicles to queue up on
either side of the intersection, leading to a significant reduction in the
average speed of vehicles in the network.
"""
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs.loop.loop_accel import ADDITIONAL_ENV_PARAMS
from flow.scenarios.figure_eight import ADDITIONAL_NET_PARAMS

vehicles = VehicleParams()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=1.5,
    ),
    initial_speed=0,
    num_vehicles=14)


flow_params = dict(
    # name of the experiment
    exp_tag='figure8',

    # name of the flow environment the experiment is running on
    env_name='AccelEnv',

    # name of the scenario class the experiment is running on
    scenario='Figure8Scenario',

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=1500,
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,
)
