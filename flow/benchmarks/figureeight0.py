"""Benchmark for figureeight0.

Trains a fraction of vehicles in a ring road structure to regulate the flow of
vehicles through an intersection. In this example, the last vehicle in the
network is an autonomous vehicle.

- **Action Dimension**: (1, )
- **Observation Dimension**: (28, )
- **Horizon**: 1500 steps
"""

from copy import deepcopy
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.scenarios.figure_eight import ADDITIONAL_NET_PARAMS

# time horizon of a single rollout
HORIZON = 1500

# We place 1 autonomous vehicle and 13 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=1.5,
    ),
    num_vehicles=13)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag="figure_eight_0",

    # name of the flow environment the experiment is running on
    env_name="AccelEnv",

    # name of the scenario class the experiment is running on
    scenario="Figure8Scenario",

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": 20,
            "max_accel": 3,
            "max_decel": 3,
            "sort_vehicles": False
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params=deepcopy(ADDITIONAL_NET_PARAMS),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)
