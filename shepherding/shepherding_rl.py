'''
RL-based shepherding. Attempts to use multiple RL vehicles to shepherd an aggressive driver (either a trained policy
or a SUMO based policy) in a multilane ring road.
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

    aggressive_cfm_params = SumoCarFollowingParams(carFollowModel="IDM", speedFactor=2, tau=0.2, minGap=1.0, accel=8)
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

    additional_net_params = {"length": 400, "lanes": 3, "speed_limit": 15, "resolution": 40}
    net_params = NetParams(additional_params=additional_net_params)

    additional_init_config_params= {"rl_out_front": False}
    initial_config = InitialConfig(spacing="uniform_in_lane", lanes_distribution=3, bunching=30, shuffle=True, additional_params=additional_init_config_params)

    scenario = LoopScenario("3-lane-aggressive-driver", ShepherdingGenerator, vehicles, net_params, initial_config)
    env_name = "ShepherdingEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params, initial_config, scenario)
    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianGRUPolicy(
        env_spec=env.spec,
        hidden_sizes=(32,)
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


for seed in [900, 1200, 1717, 2018]:
    run_experiment_lite(
        run_task,
        snapshot_mode="gap",
        snapshot_gap=50,
        exp_prefix="_shepherding_full_aggro_headways",
        # n_parallel=8,
        seed=seed,
        # mode="ec2",
    )
