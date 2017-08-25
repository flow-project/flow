"""
(document here)
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
from cistar_dev.controllers.lane_change_controllers import *
from cistar_dev.controllers.car_following_models import *

logging.basicConfig(level=logging.INFO)


num_cars = 15
num_auto = 1

exp_tag = str(num_cars) + '-car-' + str(num_auto) + '-rl-moving-bottleneck'


def run_task(*_):
    logging.basicConfig(level=logging.INFO)

    sumo_params = {"time_step": 0.1, "starting_position_shuffle": False, "vehicle_arrangement_shuffle": False,
                   "rl_lc": "no_lat_collide", "human_lc": "strategic", "rl_sm": "no_collide", "human_sm": "no_collide"}
    sumo_binary = "sumo-gui"

    env_params = {"target_velocity": 3, "max-deacc": -6, "max-acc": 3, "lane_change_duration": 0,
                  "observation_vel_std": 0, "observation_pos_std": 0, "human_acc_std": 0.5, "rl_acc_std": 0,
                  "num_steps": 1500}

    net_params = {"length": 230, "lanes": 2, "speed_limit": 30, "resolution": 40,
                  "net_path": "debug/net/"}

    cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/rl/cfg/"}

    initial_config = {"shuffle": False, "bunching": 180, "lanes_distribution": 2}

    type_params = [("rl", num_auto, (RLController, {}), None, 0),
                   ("idm", num_cars - num_auto, (IDMController, {}), (StaticLaneChanger, {}), 0)]

    scenario = LoopScenario(exp_tag, type_params, net_params, cfg_params, initial_config=initial_config)

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
        batch_size=15000,
        max_path_length=horizon,
        n_itr=1,  # 1000
        # whole_paths=True,
        discount=0.999,
        step_size=0.01,
    )
    algo.train()

for seed in [5]:  # [16, 20, 21, 22]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        python_command="/home/aboudy/anaconda2/envs/rllab-distributed/bin/python3.5"
        # plot=True,
    )
