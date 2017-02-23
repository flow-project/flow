import logging

from cistar.core.exp import SumoExperiment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.velocity_controllers import *
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *

logging.basicConfig(level=logging.INFO)

sumo_params = {"port": 8873, "time_step":0.1}

sumo_binary = "sumo-gui"

type_params = {"cfm_slow": (5, make_better_cfm(v_des=8), stochastic_lane_changer(prob = 0.5)),"cfm_fast": (5, make_better_cfm(v_des=15), stochastic_lane_changer(prob = 0.5))}

env_params = {"target_velocity": 25}

net_params = {"length": 200, "lanes": 2, "speed_limit":35, "resolution": 40, "net_path":"debug/net/"}

cfg_params = {"start_time": 0, "end_time":3000, "cfg_path":"debug/cfg/"}

initial_config = {"shuffle":False}

scenario = LoopScenario("test-exp", type_params, net_params, cfg_params, initial_config)
##data path needs to be relative to cfg location
leah_sumo_params = {"port": 8873}

exp = SumoExperiment(SimpleAccelerationEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1000)

exp.env.terminate()
