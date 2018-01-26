"""
Used as example of sugiyama experiment.
22 IDM cars on a ring create shockwaves.
"""
import logging

from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import IDMController

from flow.envs.loop_accel import AccelEnv
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario


def sugiyama_example(sumo_binary=None):
    logging.basicConfig(level=logging.INFO)

    sumo_params = SumoParams(sim_step=0.1, sumo_binary="sumo-gui")

    if sumo_binary is not None:
        sumo_params.sumo_binary = sumo_binary

    vehicles = Vehicles()
    vehicles.add(veh_id="idm",
                 acceleration_controller=(IDMController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=22)

    additional_env_params = {"target_velocity": 8}
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = {"length": 230, "lanes": 1, "speed_limit": 30,
                             "resolution": 40}
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(bunching=20)

    scenario = LoopScenario(name="sugiyama",
                            generator_class=CircleGenerator,
                            vehicles=vehicles,
                            net_params=net_params,
                            initial_config=initial_config)

    env = AccelEnv(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    logging.info("Experiment Set Up complete")

    return exp


if __name__ == "__main__":
    # import the experiment variable
    exp = sugiyama_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
