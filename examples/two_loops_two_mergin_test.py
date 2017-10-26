''' Example of figure 8 with car following models on it.


Variables:
    sumo_params {dict} -- [Pass time step, where to save data, whether safe mode is on or off]
    sumo_binary {str} -- [Use either sumo-gui or sumo for visual or non-visual]
    type_params {dict} -- [Types of cars in the system.
    Format {"name": (number, (Model, {params}), (Lane Change Model, {params}), initial_speed)}]
    env_params {dict} -- []
    net_params {dict} -- [Params for network.
                            radius_ring
                            lanes
                            speed limit
                            resolution: number of edges comprising ring
                            net_path: where to store net]
    cfg_params {dict} -- [description]
    initial_config {dict} -- [shuffle: randomly reorder cars to start experiment
                                spacing: if gaussian, add noise in start positions
                                bunching: how close to place cars at experiment start]
    scenario {[type]} -- [Which road network to use]
'''
import logging
import numpy as np

from flow.core.experiment import SumoExperiment
from flow.core.vehicles import Vehicles
from flow.core.params import EnvParams, NetParams, InitialConfig, SumoParams

from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.controllers.routing_controllers import *


from flow.envs.loop_accel import SimpleAccelerationEnvironment
from flow.scenarios.two_loops_two_merging.gen import TwoLoopTwoMergingGenerator
from flow.scenarios.two_loops_two_merging.two_loops_two_merging_scenario import TwoLoopsTwoMergingScenario

logging.basicConfig(level=logging.INFO)

sumo_binary = "sumo-gui"
sumo_params = SumoParams(sumo_binary=sumo_binary)

vehicles = Vehicles()
vehicles.add_vehicles("idm", (IDMController, {}), (StaticLaneChanger, {}), (ContinuousRouter, {}), 0, 15)

env_params = EnvParams()

additional_net_params = {"ring_radius": 30, "lanes": 1, "speed_limit": 30, "resolution": 40}
net_params = NetParams( no_internal_links=False, additional_params=additional_net_params)

initial_config = InitialConfig(distribution_length=8 / 3 * np.pi * additional_net_params["ring_radius"])

scenario = TwoLoopsTwoMergingScenario("two-loop-two-merging", TwoLoopTwoMergingGenerator, vehicles,
                                      net_params, initial_config)

env = SimpleAccelerationEnvironment(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()
