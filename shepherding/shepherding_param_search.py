import logging
from shepherding_env import ShepherdingEnv
from flow.core.experiment import SumoExperiment
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.controllers.routing_controllers import *
from flow.core.vehicles import Vehicles
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, SumoCarFollowingParams, SumoLaneChangeParams

logging.basicConfig(level=logging.INFO)

def compare_params(time_step, num_cars, lanes, length, speed_limit):
    sumo_params = SumoParams(time_step=time_step, sumo_binary="sumo")

    human_cfm_params = SumoCarFollowingParams(sigma=1.0, tau=2.0, speedDev=0.1)
    aggressive_cfm_params = SumoCarFollowingParams(speedFactor=1.75, decel=7.5, accel=4.5, tau=0.1)

    human_lc_params = SumoLaneChangeParams(lcKeepRight=0, lcAssertive=0.5, lcSpeedGain=1.5, lcSpeedGainRight=1.0, model="SL2015")
    aggressive_lc_params = SumoLaneChangeParams(lcAssertive=20, lcPushy=0.8, lcSpeedGain=100.0, lcAccelLat=6,
                                            lcSpeedGainRight=1.0, model="SL2015")

    aggressive_vehicles = Vehicles()
    aggressive_vehicles.add_vehicles("human",
                          (SumoCarFollowingController, {}),
                          (SumoLaneChangeController, {}),
                          (ContinuousRouter, {}),
                          0, num_cars - 1,
                          lane_change_mode="custom", custom_lane_change_mode=0b1000010101,
                          sumo_car_following_params=human_cfm_params,
                          sumo_lc_params=human_lc_params)
    aggressive_vehicles.add_vehicles("aggressive-human",
                          (SumoCarFollowingController, {}),
                          (SumoLaneChangeController, {}),
                          (ContinuousRouter, {}),
                          0, 1,
                          lane_change_mode="custom", custom_lane_change_mode=0b0100010101,
                          sumo_car_following_params=aggressive_cfm_params,
                          sumo_lc_params=aggressive_lc_params)

    just_human_vehicles = Vehicles()
    just_human_vehicles.add_vehicles("human",
                          (SumoCarFollowingController, {}),
                          (SumoLaneChangeController, {}),
                          (ContinuousRouter, {}),
                          0, num_cars,
                          lane_change_mode="custom", custom_lane_change_mode=0b1000010101,
                          sumo_car_following_params=human_cfm_params,
                          sumo_lc_params=human_lc_params)

    env_params = EnvParams(additional_params={"target_velocity": speed_limit})

    additional_net_params = {"length": length,
                             "lanes": lanes,
                             "speed_limit": speed_limit,
                             "resolution": 40}
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform_random", scale=8, x0=20, lanes_distribution=3, shuffle=True)

    aggressive_scenario = LoopScenario("single-lane-two-contr", CircleGenerator, aggressive_vehicles, net_params, initial_config)
    aggressive_reward = SumoExperiment(ShepherdingEnv(env_params, sumo_params, aggressive_scenario), aggressive_scenario).run(3, 1000)

    just_human_scenario = LoopScenario("single-lane-two-contr", CircleGenerator, just_human_vehicles, net_params, initial_config)
    just_human_reward = SumoExperiment(ShepherdingEnv(env_params, sumo_params, just_human_scenario), just_human_scenario).run(3, 1000)

    return aggressive_reward, just_human_reward

aggro_param_reward_map = {}
human_param_reward_map = {}
difference_param_reward_map = {}

for time_step in [0.1]:
    for num_cars in [24, 36, 48, 60]:
        for lanes in [3, 4]:
            for length in [300, 500, 800]:
                for speed_limit in [15]:
                        aggressive_reward, just_human_reward = compare_params(time_step, num_cars, lanes, length, speed_limit)
                        aggro_param_reward_map[(time_step, num_cars, lanes, length, speed_limit)] = aggressive_reward
                        human_param_reward_map[
                            (time_step, num_cars, lanes, length, speed_limit)] = just_human_reward
                        difference_param_reward_map[
                            (time_step, num_cars, lanes, length, speed_limit)] = aggressive_reward - just_human_reward
                        print((time_step, num_cars, lanes, length, speed_limit), aggressive_reward, just_human_reward, aggressive_reward - just_human_reward)

print(aggro_param_reward_map)
print(human_param_reward_map)
print(difference_param_reward_map)