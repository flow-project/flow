"""
Example of a figure 8 network with human-driven vehicles.
"""
import logging
from flow.core.params import SumoParams, EnvParams, NetParams
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.vehicles import Vehicles

from flow.core.experiment import SumoExperiment
from flow.envs.loop_accel import AccelEnv
from flow.scenarios.figure8.gen import Figure8Generator
from flow.scenarios.figure8.figure8_scenario import Figure8Scenario, \
    ADDITIONAL_NET_PARAMS
from flow.controllers.car_following_models import IDMController
from flow.controllers.lane_change_controllers import StaticLaneChanger


def figure_eight_example(sumo_binary=None):

    logging.basicConfig(level=logging.INFO)

    sumo_params = SumoParams(sumo_binary="sumo-gui")

    if sumo_binary is not None:
        sumo_params.sumo_binary = sumo_binary

    vehicles = Vehicles()
    vehicles.add(veh_id="idm",
                 acceleration_controller=(IDMController, {}),
                 lane_change_controller=(StaticLaneChanger, {}),
                 routing_controller=(ContinuousRouter, {}),
                 initial_speed=0,
                 num_vehicles=14)

    additional_env_params = {"target_velocity": 8, "num_steps": 500}
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(no_internal_links=False,
                           additional_params=additional_net_params)

    scenario = Figure8Scenario(name="figure8",
                               generator_class=Figure8Generator,
                               vehicles=vehicles,
                               net_params=net_params)

    env = AccelEnv(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    logging.info("Experiment Set Up complete")

    return exp


if __name__ == "__main__":

    # import the experiment variable
    exp = figure_eight_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
