""" Example of ring road with larger merging ring.

"""
import logging
import numpy as np

from cistar.core.experiment import SumoExperiment
from cistar.core.vehicles import Vehicles

from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.routing_controllers import *

from cistar.envs.two_loops_one_merging import SimpleAccelerationEnvironment
from cistar.scenarios.two_loops_one_merging.gen import TwoLoopOneMergingGenerator
from cistar.scenarios.two_loops_one_merging.two_loops_one_merging_scenario import TwoLoopsOneMergingScenario

logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step": 0.1, "emission_path": "./data/", "human_sm": 1}

sumo_binary = "sumo-gui"

# note that the vehicles are added sequentially by the generator,
# so place the merging vehicles after the vehicles in the ring
vehicles = Vehicles()
vehicles.add_vehicles("idm", (IDMController, {}), (StaticLaneChanger, {}), (ContinuousRouter, {}), 0, 11)
vehicles.add_vehicles("merge-idm", (IDMController, {}), (StaticLaneChanger, {}), (ContinuousRouter, {}), 0, 11)

env_params = {"max-deacc": -3, "max-acc": 3}

net_params = {"ring_radius": 230/(2*np.pi), "lanes": 1, "speed_limit": 30, "resolution": 40,
              "net_path": "debug/net/", "no-internal-links": False}

cfg_params = {"start_time": 0, "end_time": 3000, "cfg_path": "debug/cfg/"}

initial_config = {"spacing": "custom"}

scenario = TwoLoopsOneMergingScenario("two-loop-one-merging", TwoLoopOneMergingGenerator, vehicles,
                                      net_params, cfg_params, initial_config)

env = SimpleAccelerationEnvironment(env_params, sumo_binary, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 3000)

exp.env.terminate()
