import logging

from cistar.core.experiment import SumoExperiment
from cistar.envs.loop_with_perturbation import PerturbationAccelerationLoop
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.car_following_models import *
from cistar.controllers.velocity_controllers import *
from cistar.controllers.lane_change_controllers import *

logging.basicConfig(level=logging.INFO)

sumo_params = {"port": 8873, "time_step":0.01, "emission_path": "./test_time_rollout"}

sumo_binary = "sumo-gui"

# type_params = {"ovm": (12, (OVMController, {'h_go': 6}), never_change_lanes_controller(), 0), 
# 			   'const': (2, (ConstantVelocityController, {'constant_speed': 28}), never_change_lanes_controller(), 0)}
# # h_go = 11.8 with OVM: gets you 8.29 m/s
# # h_go = 13.2 with OVM: gets you 6.48 m/s

type_params = {"ovm": (22, (OVMController, {'v_max': 15}), (StaticLaneChanger, {}), 0)}

env_params = {"target_velocity": 25, "perturbations": [(500, 150), (1000, 150), (1500, 150)],
              "max-deacc": -5, "max-acc": 5, 'fail-safe': 'instantaneous'}

net_params = {"length": 230, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"debug/net/"}

cfg_params = {"start_time": 0, "end_time":3000, "cfg_path":"debug/cfg/"}

initial_config = {"shuffle": False}

scenario_name = 'sugiyama-perturbation-ovm'

scenario = LoopScenario(scenario_name, type_params, net_params, cfg_params, initial_config)

leah_sumo_params = {"port": 8873}

env = PerturbationAccelerationLoop(env_params, sumo_binary, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 10000)

exp.env.terminate()

