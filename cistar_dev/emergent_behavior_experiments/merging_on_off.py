"""
"""

import logging
import sys

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from cistar_dev.scenarios.loop_merges.loop_merges_scenario import LoopMergesScenario
from cistar_dev.controllers.rlcontroller import RLController
from cistar_dev.controllers.lane_change_controllers import *
from cistar_dev.controllers.car_following_models import *

from numpy import pi


def run_task(v):
    import cistar_dev.envs as cistar_envs
    logging.basicConfig(level=logging.INFO)

    sumo_params = {"time_step": 0.1, "starting_position_shuffle": False, "vehicle_arrangement_shuffle": False,
                   "rl_lc": "no_lat_collide", "human_lc": "no_lat_collide",
                   "rl_sm": "no_collide", "human_sm": "no_collide"}
    sumo_binary = "sumo"

    env_params = {"target_velocity": 8, "max-deacc": -6, "max-acc": 3, "num_steps": 1500,
                  "observation_pos_std": 0, "observation_vel_std": 0, "human_acc_std": 0, "rl_acc_std": 0}

    net_params = {"merge_in_length": 500, "merge_in_angle": pi / 9,
                  "merge_out_length": 500, "merge_out_angle": pi * 17 / 9,
                  "ring_radius": 400 / (2 * pi), "resolution": 40, "lanes": 1, "speed_limit": 30,
                  "net_path": "debug/net/"}

    cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/rl/cfg/"}

    initial_config = {"merge_bunching": 200, "n_merge_platoons": 2}

    num_merge = 14
    num_non_merge = 14
    num_auto = 1
    exp_tag = str(num_merge + num_non_merge) + "-car-" + str(num_merge) + "-merge-" + str(num_auto) + "-rl-merge-on-off"

    type_params = [("rl", num_auto, (RLController, {}), None, 0),
                   ("idm", num_non_merge - num_auto, (IDMController, {}), None, 0),
                   ("merge-idm", num_merge, (IDMController, {}), None, 0)]

    scenario = LoopMergesScenario(exp_tag, type_params, net_params, cfg_params, initial_config=initial_config)

    env_name = "SimpleLoopMergesEnvironment"
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
        n_itr=1000,
        # whole_paths=True,
        discount=0.999,
        # step_size=v["step_size"],
    )
    algo.train(),

num_merge = 14
num_non_merge = 14
num_auto = 1
exp_tag = str(num_merge + num_non_merge) + "-car-" + str(num_merge) + "-merge-" + str(num_auto) + "-rl-merge-on-off"

for seed in [5, 20]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=8,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="ec2",
        exp_prefix=exp_tag,
        # python_command="/home/aboudy/anaconda2/envs/rllab-distributed/bin/python3.5"
        # plot=True,
    )
