import logging

from cistar.core.exp import SumoExperiment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.figure8.figure8_scenario import Figure8Scenario
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.rlcontroller import RLController

logging.basicConfig(level=logging.INFO)

sumo_params = {"port": 8873, "time_step": 0.1, "emission_path": "./data/", "traci_control": 1,
               "rl_lc": "aggressive", "human_lc": "aggressive", "rl_sm": "aggressive", "human_sm": "aggressive"}

sumo_binary = "sumo-gui"

type_params = {
    "dd":  (1, (DrunkDriver, {}), (StaticLaneChanger, {}), 0),
    "idm": (21, (IDMController, {}), (StaticLaneChanger, {}), 0)}

env_params = {"target_velocity": 25, "max-deacc": -3, "max-acc": 3, "fail-safe": "None",
              "intersection_fail-safe": "None"}

radius_ring = 30
net_params = {"radius_ring": radius_ring, "lanes": 1, "speed_limit": 30, "resolution": 40,
              "net_path": "debug/net/", "length": 230}

cfg_params = {"start_time": 0, "end_time": 3000, "cfg_path": "debug/cfg/"}

initial_config = {"shuffle": False}

scenario = LoopScenario("loop", type_params, net_params, cfg_params, initial_config=initial_config)

leah_sumo_params = {"port": 8873}

exp = SumoExperiment(SimpleAccelerationEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()
