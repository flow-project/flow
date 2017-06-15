import logging

from cistar.core.exp import SumoExperiment
from cistar.envs.loop_accel_emission import SimpleAccelerationEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *

logging.basicConfig(level=logging.INFO)

sumo_params = {"time_step":0.1, "traci_control": 1, "human_sm": "no_collide"}

sumo_binary = "sumo-gui"

type_params = {"idm": (15, (IDMController, {}), (StaticLaneChanger, {}), 0), 
                "idm2": (1, (DrunkDriver, {}), (StaticLaneChanger, {}), 0)}
#type_params = {"idm": (16, (IDMController, {"a":5.0, "b":3.0, "T":.5, "v0":50}), (StaticLaneChanger, {}), 0)}

env_params = {"target_velocity": 25}

net_params = {"length": 200, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"debug/net/"}

cfg_params = {"start_time": 0, "end_time":3000, "cfg_path":"debug/cfg/"}


initial_config = {"shuffle": True, "spacing":"gaussian"}

scenario = LoopScenario("single-lane-one-contr", type_params, net_params, cfg_params, initial_config)
##data path needs to be relative to cfg location

exp = SumoExperiment(SimpleAccelerationEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

exp.run(400, 1000)

exp.env.terminate()
