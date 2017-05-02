import logging

from cistar.core.exp import SumoExperiment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.envs.loop_velocity import SimpleVelocityEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.velocity_controllers import *


env_params = {"target_velocity" : 8}

sumo_binary = "sumo-gui"

sumo_params = {"port" : 8873, "time_step" : 0.1}

# type_params = { "cfm-slow": (6, (CFMController, {'v_des': 6}), never_change_lanes_controller(), 0),\
#  "cfm-fast": (6, (CFMController, {'v_des': 10}), never_change_lanes_controller(), 0)}

# type_params = { "cfm-slow": (6, (CFMController, {'v_des': 3}), (StaticLaneChanger, {}), 0),\
#  "cfm-fast": (6, (CFMController, {'v_des': 20}), (StochasticLaneChanger, {}), 0)}

type_params = { "cfm-slow": (6, (CFMController, {'v_des': 3}), None, 0),\
 "cfm-fast": (6, (CFMController, {'v_des': 20}), None, 0)}

# type_params = { "cfm-slow": (6, (ConstantVelocityController, {}), None, 0),\
#  "cfm-fast": (6, (ConstantVelocityController, {}), None, 0)}

net_params = {"length": 200, "lanes": 2, "speed_limit":35,\
 "resolution": 40, "net_path":"debug/net/"}

cfg_params = {"start_time": 0, "end_time":3000, "cfg_path":"debug/cfg/"}

scenario = LoopScenario("two-lane-two-controller", type_params, net_params, cfg_params)

# exp = SumoExperiment(SimpleVelocityEnvironment, env_params,\
#  sumo_binary, sumo_params, scenario)

exp = SumoExperiment(SimpleAccelerationEnvironment, env_params,\
 sumo_binary, sumo_params, scenario)


exp.run(2, 10000)