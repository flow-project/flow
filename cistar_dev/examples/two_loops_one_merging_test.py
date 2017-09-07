""" Example of ring road with larger merging ring.

"""
import logging
import numpy as np

from cistar.core.experiment import SumoExperiment
from cistar.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from cistar.core.vehicles import Vehicles

from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.routing_controllers import *

from cistar.envs.two_loops_one_merging import SimpleAccelerationEnvironment
from cistar.scenarios.two_loops_one_merging.gen import TwoLoopOneMergingGenerator
from cistar.scenarios.two_loops_one_merging.two_loops_one_merging_scenario import TwoLoopsOneMergingScenario

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(time_step=0.1, emission_path="./data/", human_speed_mode="no_collide",
                         sumo_binary="sumo-gui")

# note that the vehicles are added sequentially by the generator,
# so place the merging vehicles after the vehicles in the ring
vehicles = Vehicles()
vehicles.add_vehicles("idm", (IDMController, {}), (StaticLaneChanger, {}), (ContinuousRouter, {}), 0, 11)
vehicles.add_vehicles("merge-idm", (IDMController, {}), (StaticLaneChanger, {}), (ContinuousRouter, {}), 0, 11)

additional_env_params = {"target_velocity": 8, "max-deacc": -6, "max-acc": 3}
env_params = EnvParams(additional_params=additional_env_params)

additional_net_params = {"ring_radius": 230/(2*np.pi), "lanes": 1, "speed_limit": 30, "resolution": 40}
net_params = NetParams(no_internal_links=False, additional_params=additional_net_params)

initial_config = InitialConfig(spacing="custom")

scenario = TwoLoopsOneMergingScenario("two-loop-one-merging", TwoLoopOneMergingGenerator, vehicles,
                                      net_params, initial_config)

env = SimpleAccelerationEnvironment(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 3000)

exp.env.terminate()
