"""
Example of a figure 8 network with human-driven vehicles.
"""
import logging
from flow.core.params import SumoParams, EnvParams, NetParams
from flow.controllers.routing_controllers import *
from flow.core.vehicles import Vehicles

from flow.core.experiment import SumoExperiment
from flow.envs.loop_accel import SimpleAccelerationEnvironment
from flow.scenarios.figure8.gen import Figure8Generator
from flow.scenarios.figure8.figure8_scenario import Figure8Scenario
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(sumo_binary="sumo-gui")

vehicles = Vehicles()
vehicles.add(veh_id="idm",
             acceleration_controller=(IDMController, {}),
             lane_change_controller=(StaticLaneChanger, {}),
             routing_controller=(ContinuousRouter, {}),
             initial_speed=0,
             num_vehicles=14)

additional_env_params = {"target_velocity": 8, "num_steps": 500}
env_params = EnvParams(additional_params=additional_env_params)

additional_net_params = {"radius_ring": 30, "lanes": 1,
                         "speed_limit": 30, "resolution": 40}
net_params = NetParams(no_internal_links=False,
                       additional_params=additional_net_params)

scenario = Figure8Scenario(name="figure8",
                           generator_class=Figure8Generator,
                           vehicles=vehicles,
                           net_params=net_params)

env = SimpleAccelerationEnvironment(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)

exp.env.terminate()
