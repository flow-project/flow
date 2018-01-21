'''
Test bed for seeing the effects of a SUMO based aggressive driver in a multilane ring road experiment.
'''

from flow.envs.shepherding_env import ShepherdingEnv
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.controllers.routing_controllers import *
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.vehicles import Vehicles
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(time_step= 0.1, sumo_binary="sumo-gui")

vehicles = Vehicles()
human_cfm_params = SumoCarFollowingParams(carFollowModel="IDM", tau=3.0, speedDev=0.1, minGap=1.0)
human_lc_params = SumoLaneChangeParams(lcKeepRight=0, lcAssertive=0.5,
                                       lcSpeedGain=1.5, lcSpeedGainRight=1.0)
vehicles.add_vehicles("human", (SumoCarFollowingController, {}), (SumoLaneChangeController, {}),
                      (ContinuousRouter, {}),
                      0, 10,
                      lane_change_mode="execute_all",
                      sumo_car_following_params=human_cfm_params,
                      sumo_lc_params=human_lc_params,
                      )

aggressive_cfm_params = SumoCarFollowingParams(carFollowModel="IDM", speedFactor=2, tau=0.2, minGap=1.0, accel=4.5)
vehicles.add_vehicles("aggressive-human", (SumoCarFollowingController, {}),
                      (SafeAggressiveLaneChanger, {"target_velocity": 22.25, "threshold": 0.8}),
                      (ContinuousRouter, {}), 0, 1,
                      lane_change_mode="custom", custom_lane_change_mode=0b0100000000,
                      sumo_car_following_params=aggressive_cfm_params)

env_params = EnvParams(additional_params={"target_velocity": 15, "num_steps": 1000}, lane_change_duration=0.1,
                       max_speed=30)

additional_net_params = {"length": 300, "lanes": 3, "speed_limit": 15, "resolution": 40}
net_params = NetParams(additional_params=additional_net_params)

initial_config = InitialConfig(spacing="custom", lanes_distribution=2, shuffle=True)

scenario = LoopScenario("single-lane-two-contr", CircleGenerator, vehicles, net_params,
                        initial_config)
# data path needs to be relative to cfg location

env = ShepherdingEnv(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

avg_reward = exp.run(1, 15000)

print(avg_reward)

exp.env.terminate()
