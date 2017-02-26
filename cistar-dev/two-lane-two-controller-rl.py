import logging

from cistar.core.exp import SumoExperiment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *


env_params = {"target_velocity" : 8}

sumo_binary = "sumo-gui"

sumo_params = {"port" : 8873, "time_step" : 0.1}

type_params = { "cfm-slow": (6, make_better_cfm(v_des=6),  never_change_lanes_controller()),\
 "cfm-fast": (6, make_better_cfm(v_des=10), never_change_lanes_controller())}

net_params = {"length": 200, "lanes": 1, "speed_limit":35,\
 "resolution": 40, "net_path":"debug/net/"}

cfg_params = {"start_time": 0, "end_time":3000, "cfg_path":"debug/cfg/"}

scenario = LoopScenario("tutorial-experiment", type_params, net_params, cfg_params)

exp = SumoExperiment(SimpleAccelerationEnvironment, env_params,\
 sumo_binary, sumo_params, scenario)

exp.run(2, 1000)