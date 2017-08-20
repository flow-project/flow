"""
Script used to test the incorporation of lane changing into rl vehicles.
Vehicles are trained to travel at a target velocity while avoiding collisions.
"""

import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from cistar_dev.scenarios.loop.loop_scenario import LoopScenario
from cistar_dev.controllers.rlcontroller import RLController

num_cars = 2
exp_tag = str(num_cars) + '-car-rl-lane_change'


def run_task(*_):
    import cistar_dev.envs as cistar_envs
    logging.basicConfig(level=logging.INFO)

    sumo_params = {"time_step": 0.1, "rl_lc": "no_lat_collide", "human_lc": "no_collide",
                   "rl_sm": "no_collide", "human_sm": "no_collide"}
    sumo_binary = "sumo"

    initial_config = {"shuffle": False}

    type_params = [("rl", num_cars, (RLController, {}), None, 0)]

    env_params = {"target_velocity": 8, "max-deacc": -6, "max-acc": 3, "lane_change_duration": 5,
                  "fail-safe": "None", "num_steps": 1500}

    net_params = {"length": 230, "lanes": 2, "speed_limit": 30, "resolution": 40, "net_path": "debug/net/"}

    cfg_params = {"start_time": 0, "end_time": 30000000, "cfg_path": "debug/cfg/"}

    scenario = LoopScenario("two-lane-two-controller", type_params, net_params, cfg_params, initial_config)

    from cistar_dev import pass_params
    env_name = "SimpleLaneChangingAccelerationEnvironment"
    pass_params = (env_name, sumo_params, sumo_binary, type_params, env_params, net_params,
                   cfg_params, initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=30000,
        max_path_length=horizon,
        n_itr=2,  # 50000
        # whole_paths=True,
        discount=0.99,
        # step_size=0.01,
    )
    algo.train(),


for seed in [5]:  # [5, 10, 73, 56, 1]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=4,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        # python_command="/home/aboudy/anaconda2/envs/rllab3/bin/python3.5"
        # plot=True,
    )
