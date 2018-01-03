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
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.controllers.rlcarfollowingcontroller import RLCarFollowingController
from flow.controllers.routing_controllers import *
from flow.core.params import *
from flow.core.vehicles import Vehicles
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.scenarios.shepherding.shepherding_generator import ShepherdingGenerator

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy


def run_task(*_):

    sumo_params = SumoParams(time_step=0.1, sumo_binary="sumo-gui")

    vehicles = Vehicles()

    # human_cfm_params = SumoCarFollowingParams(carFollowModel="IDM", tau=3.0, speedDev=0.1, minGap=1.0)
    # human_lc_params = SumoLaneChangeParams(lcKeepRight=0, lcAssertive=0.5,
    #                                        lcSpeedGain=1.5, lcSpeedGainRight=1.0)
    # vehicles.add_vehicles("human", (SumoCarFollowingController, {}), (SumoLaneChangeController, {}),
    #                       (ContinuousRouter, {}),
    #                       0, 10,
    #                       lane_change_mode="execute_all",
    #                       sumo_car_following_params=human_cfm_params,
    #                       sumo_lc_params=human_lc_params,
    #                       )

    aggressive_cfm_params = SumoCarFollowingParams(carFollowModel="IDM", speedFactor=2, tau=0.2, minGap=1.0, accel=4.5)
    vehicles.add_vehicles("aggressive-human", (SumoCarFollowingController, {}),
                          (SafeAggressiveLaneChanger, {"target_velocity": 22.25, "threshold": 0.8}),
                          (ContinuousRouter, {}), 0, 1,
                          lane_change_mode="custom", custom_lane_change_mode=0b0100000000,
                          sumo_car_following_params=aggressive_cfm_params)


    rl_cfm_params = SumoCarFollowingParams(carFollowModel="IDM", tau=1.0, speedDev=2, minGap=1.0)
    vehicles.add_vehicles("rl", (RLCarFollowingController, {}), None, (ContinuousRouter, {}), 0, 3,
                          lane_change_mode="custom", custom_lane_change_mode=512,
                          sumo_car_following_params=rl_cfm_params, additional_params={"emergencyDecel":"9"})

    env_params = EnvParams(additional_params={"target_velocity": 15, "num_steps": 1000},
                           lane_change_duration=0.1, max_speed=30)

    additional_net_params = {"length": 800, "lanes": 3, "speed_limit": 15, "resolution": 40}
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="custom", bunching=750, lanes_distribution=3, shuffle=False)

    # scenario = LoopScenario("3-lane-aggressive-driver", CircleGenerator, vehicles, net_params, initial_config)
    scenario = LoopScenario("3-lane-aggressive-driver", ShepherdingGenerator, vehicles, net_params, initial_config)
    env_name = "ShepherdingEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params, initial_config, scenario)
    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianGRUPolicy(
        env_spec=env.spec,
        hidden_sizes=(16,)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=30000,
        max_path_length=horizon,
        n_itr=1001,
    )
    algo.train()


for seed in [50, 55, 60, 65, 70]:
    run_experiment_lite(
        run_task,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=50,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used,
        exp_prefix="shepherding_base_case_big_loop_1k_itr_hidden_size_16",
        # Number of parallel workers for sampling
        n_parallel=1,
        seed=seed,
        # python_command="/Users/kanaad/anaconda3/envs/flow/bin/python",
        mode="local",
        # n_parallel=1,
    )
