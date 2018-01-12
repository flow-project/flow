"""
This script presents the use of two-way intersections in flow.

Cars enter from the bottom and left nodes following a probability distribution,
and continue to move straight until they exit through the top and right nodes,
respectively.
"""
from flow.core.vehicles import Vehicles
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams

from flow.core.experiment import SumoExperiment
from flow.envs.two_intersection import TwoIntersectionEnv
from flow.scenarios.intersections.gen import TwoWayIntersectionGenerator
from flow.scenarios.intersections.intersection_scenario import *
from flow.controllers.car_following_models import *

import logging


def two_way_intersection_example(sumo_binary=None):

    logging.basicConfig(level=logging.INFO)

    sumo_params = SumoParams(sim_step=0.1, emission_path="./data/",
                             sumo_binary="sumo-gui")

    if sumo_binary is not None:
        sumo_params.sumo_binary = sumo_binary

    vehicles = Vehicles()
    vehicles.add(veh_id="idm",
                 acceleration_controller=(IDMController, {}),
                 num_vehicles=20)

    intensity = .2
    v_enter = 10

    env_params = EnvParams(additional_params={"target_velocity": v_enter,
                                              "control-length": 150,
                                              "max_speed": v_enter})

    additional_net_params = \
        {"horizontal_length_in": 400, "horizontal_length_out": 10, "horizontal_lanes": 1,
         "vertical_length_in": 400, "vertical_length_out": 10, "vertical_lanes": 1,
         "speed_limit": {"horizontal": 30, "vertical": 30}}
    net_params = NetParams(no_internal_links=False,
                           additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="custom",
                                   additional_params={"intensity": intensity,
                                                      "enter_speed": v_enter})

    scenario = TwoWayIntersectionScenario(
        name="two-way-intersection",
        generator_class=TwoWayIntersectionGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config
    )

    env = TwoIntersectionEnv(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    logging.info("Experiment Set Up complete")

    return exp


if __name__ == "__main__":

    # import the experiment variable
    exp = two_way_intersection_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
