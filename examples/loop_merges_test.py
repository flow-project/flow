"""
(description)
"""

import logging
from flow.core.experiment import SumoExperiment
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.envs.loop_merges import SimpleLoopMergesEnvironment
from flow.scenarios.loop_merges.gen import LoopMergesGenerator
from flow.scenarios.loop_merges.loop_merges_scenario import LoopMergesScenario

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.core.experiment import SumoExperiment

from flow.envs.loop_merges import SimpleLoopMergesEnvironment
from flow.scenarios.loop_merges.gen import LoopMergesGenerator
from flow.scenarios.loop_merges.loop_merges_scenario import LoopMergesScenario

from numpy import pi

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(time_step=0.1, emission_path="./data/", sumo_binary="sumo-gui")

vehicles = Vehicles()
vehicles.add_vehicles("idm", (IDMController, {}), (StaticLaneChanger, {}), None, 0, 14)
vehicles.add_vehicles("merge-idm", (IDMController, {}), (StaticLaneChanger, {}), None, 0, 14)

additional_env_params = {"target_velocity": 8, "fail-safe": "None"}
env_params = EnvParams(additional_params=additional_env_params)

additional_net_params = {"merge_in_length": 500, "merge_in_angle": pi/9,
                         "merge_out_length": 500, "merge_out_angle": pi * 17/9,
                         "ring_radius": 400 / (2 * pi), "resolution": 40, "lanes": 1, "speed_limit": 30}
net_params = NetParams(no_internal_links=False, additional_params=additional_net_params)

initial_config = InitialConfig(spacing="custom",
                               additional_params={"merge_bunching": 250})

scenario = LoopMergesScenario("loop-merges", LoopMergesGenerator, vehicles, net_params,
                              initial_config=initial_config)

env = SimpleLoopMergesEnvironment(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 550)

exp.env.terminate()
