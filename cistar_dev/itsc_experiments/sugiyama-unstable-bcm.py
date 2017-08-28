import logging

from cistar_dev.core.exp import SumoExperiment
from cistar_dev.envs.loop_accel import SimpleAccelerationEnvironment
from cistar_dev.scenarios.loop.loop_scenario import LoopScenario
from cistar_dev.controllers.car_following_models import *
from cistar_dev.controllers.lane_change_controllers import *

logging.basicConfig(level=logging.INFO)

sumo_params = {"port": 8873, "time_step":0.01}

sumo_binary = "sumo"

type_params = {"BCM": (22, (BCMController, {}), (StaticLaneChanger, {}), 0)}

env_params = {"target_velocity": 25, 'fail-safe':'eugene'}

net_params = {"length": 230, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"debug/net/"}

cfg_params = {"start_time": 0, "end_time":3000, "cfg_path":"debug/cfg/"}

initial_config = {"shuffle":False, "bunching":40, "spacing":"gaussian"}

scenario = LoopScenario("sugiyama-unstable-bcm", type_params, net_params, cfg_params, initial_config)
##data path needs to be relative to cfg location
leah_sumo_params = {"port": 8873}

env = SimpleAccelerationEnvironment(env_params, sumo_binary, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 10000)

exp.env.terminate()
