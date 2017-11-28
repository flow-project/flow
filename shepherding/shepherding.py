''' Used to test out a mixed environment with an IDM controller and
another type of car, in this case our drunk driver class. One lane. 

Variables:
    sumo_params {dict} -- [Pass time step, safe mode is on or off]
    sumo_binary {str} -- [Use either sumo-gui or sumo for visual or non-visual]
    type_params {dict} -- [Types of cars in the system. 
    Format {"name": (number, (Model, {params}), (Lane Change Model, {params}), initial_speed)}]
    env_params {dict} -- [Params for reward function]
    net_params {dict} -- [Params for network.
                            length: road length
                            lanes
                            speed limit
                            resolution: number of edges comprising ring
                            net_path: where to store net]
    cfg_params {dict} -- [description]
    initial_config {dict} -- [shuffle: randomly reorder cars to start experiment
                                spacing: if gaussian, add noise in start positions
                                bunching: how close to place cars at experiment start]
    scenario {[type]} -- [Which road network to use]
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

sumo_params = SumoParams(time_step= 0.1, sumo_binary="sumo")

vehicles = Vehicles()
human_cfm_params = SumoCarFollowingParams(carFollowModel="IDM", sigma=1.0, tau=3.0, speedDev=0.1, minGap=3.0)
human_lc_params = SumoLaneChangeParams(lcKeepRight=0, lcAssertive=0.5, lcSpeedGain=1.5, lcSpeedGainRight=1.0,
                                       model="SL2015")
vehicles.add_vehicles("human", (SumoCarFollowingController, {}), (SumoLaneChangeController, {}), (ContinuousRouter, {}),
                      0, 17,
                      lane_change_mode="execute_all",
                      sumo_car_following_params=human_cfm_params,
                      sumo_lc_params=human_lc_params,
                      )

aggressive_cfm_params = SumoCarFollowingParams(carFollowModel="IDM", speedFactor=1.75, tau=0.1, minGap=0.5)
vehicles.add_vehicles("aggressive-human", (SumoCarFollowingController, {}),
                      (SafeAggressiveLaneChanger, {"target_velocity": 22.25, "threshold": 0.7}),
                      (ContinuousRouter, {}), 0, 1,
                      lane_change_mode="custom", custom_lane_change_mode=0b0100000000,
                      sumo_car_following_params=aggressive_cfm_params)

env_params = EnvParams(additional_params={"target_velocity":15}, lane_change_duration=0)
# env_params.fail_safe = "safe_velocity"

additional_net_params = {"length": 300, "lanes": 3, "speed_limit": 15, "resolution": 40}
net_params = NetParams(additional_params=additional_net_params)

initial_config = InitialConfig(spacing="custom", lanes_distribution=3, shuffle=True)

scenario = LoopScenario("single-lane-two-contr", CircleGenerator, vehicles, net_params,
                        initial_config)
# data path needs to be relative to cfg location

env = ShepherdingEnv(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

avg_reward = exp.run(1, 15000)

exp.env.terminate()

print(avg_reward)