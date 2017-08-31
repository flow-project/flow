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

from cistar.core.experiment import SumoExperiment
from cistar.core.vehicles import Vehicles

from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.routing_controllers import *

from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.two_loops_two_merging.gen import TwoLoopTwoMergingGenerator
from cistar.scenarios.two_loops_two_merging.two_loops_two_merging_scenario import TwoLoopsTwoMergingScenario

logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step": 0.1, "emission_path": "./data/", "human_sm": 1}

sumo_binary = "sumo-gui"

vehicles = Vehicles()
vehicles.add_vehicles("idm", (IDMController, {}), (StaticLaneChanger, {}), (ContinuousRouter, {}), 0, 15)

env_params = {"max-deacc": -3, "max-acc": 3}

net_params = {"ring_radius": 30, "lanes": 1, "speed_limit": 30, "resolution": 40,
              "net_path": "debug/net/", "no-internal-links": False}

cfg_params = {"start_time": 0, "end_time": 3000, "cfg_path": "debug/cfg/"}

initial_config = {"shuffle": False, "distribution_length": 8 / 3 * np.pi * net_params["ring_radius"]}

scenario = TwoLoopsTwoMergingScenario("two-loop-two-merging", TwoLoopTwoMergingGenerator, vehicles,
                                      net_params, cfg_params, initial_config)

env = SimpleAccelerationEnvironment(env_params, sumo_binary, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()
