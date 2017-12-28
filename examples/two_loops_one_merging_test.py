"""
Example of ring road with larger merging ring.
"""
import logging
import numpy as np

from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles

from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.controllers.routing_controllers import *

from flow.envs.two_loops_one_merging import TwoLoopsOneMergingEnvironment
from flow.scenarios.two_loops_one_merging.gen import TwoLoopOneMergingGenerator
from flow.scenarios.two_loops_one_merging.two_loops_one_merging_scenario import TwoLoopsOneMergingScenario

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(sim_step=0.1, emission_path="./data/",
                         sumo_binary="sumo-gui")

# note that the vehicles are added sequentially by the generator,
# so place the merging vehicles after the vehicles in the ring
vehicles = Vehicles()
vehicles.add_vehicles(veh_id="idm",
                      acceleration_controller=(IDMController, {}),
                      routing_controller=(ContinuousRouter, {}),
                      num_vehicles=12)
vehicles.add_vehicles(veh_id="merge-idm",
                      acceleration_controller=(IDMController, {}),
                      routing_controller=(ContinuousRouter, {}),
                      num_vehicles=5)

additional_env_params = {"target_velocity": 8, "max-deacc": -6, "max-acc": 3}
env_params = EnvParams(additional_params=additional_env_params)

additional_net_params = {"ring_radius": 230/(2*np.pi), "lanes": 1,
                         "speed_limit": 30, "resolution": 40}
net_params = NetParams(
    no_internal_links=False,
    additional_params=additional_net_params
)

initial_config = InitialConfig(
    spacing="custom",
    additional_params={"merge_bunching": 0}
)

scenario = TwoLoopsOneMergingScenario(
    name="two-loop-one-merging",
    generator_class=TwoLoopOneMergingGenerator,
    vehicles=vehicles,
    net_params=net_params,
    initial_config=initial_config
)

env = TwoLoopsOneMergingEnvironment(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()
