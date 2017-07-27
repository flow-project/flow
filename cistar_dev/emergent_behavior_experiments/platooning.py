"""
Script used for teaching rl vehicles to platoon in the presence of human drivers.

The assumption is that rl vehicles are are closer together will decide to stick closer together in order to provide
human drivers with larger headways, thereby increasing their expected steady-state velocities and allowing for larger
accelerations.

Platooning is implemented by using the same techniques and reward functions as rl-lc-testing.py, but with the addition
of human drivers (modeled by an IDM controller). Moreover, more rl-vehicles are placed on the ring than human drivers,
in order to ensure that instances exist where at least two rl-vehicles are behind one another.
"""

import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from cistar_dev.envs.lane_changing import SimpleLaneChangingAccelerationEnvironment
from cistar_dev.scenarios.loop.loop_scenario import LoopScenario
from cistar_dev.controllers.rlcontroller import RLController
from cistar_dev.controllers.lane_change_controllers import *
from cistar_dev.controllers.car_following_models import *

num_cars = 44
num_auto = 22

exp_tag = str(num_cars) + '-car-' + str(num_auto) + '-rl-multi-lane-loop'

def run_task(*_):
    logging.basicConfig(level=logging.INFO)

    sumo_params = {"time_step": 0.1, "traci_control": 1, "rl_lc": "no_lat_collide", "human_lc": "strategic",
                   "rl_sm": "no_collide", "human_sm": "no_collide"}
    sumo_binary = "sumo"

    env_params = {"target_velocity": 8, "max-deacc": -6, "max-acc": 3, "lane_change_duration": 3,
                  "observation_vel_std": 0, "observation_pos_std": 0, "human_acc_std": 0, "rl_acc_std": 0,
                  "num_steps":500}

    net_params = {"length": 230, "lanes": 2, "speed_limit": 30, "resolution": 40,
                  "net_path": "debug/net/"}

    cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/rl/cfg/"}

    initial_config = {"shuffle": False}

    type_params = {"rl": (num_auto, (RLController, {}), None, 0),
                   "idm": (num_cars - num_auto, (IDMController, {}), None, 0)}

    scenario = LoopScenario(exp_tag, type_params, net_params, cfg_params, initial_config=initial_config)

    from cistar_dev import pass_params
    env_name = "SimpleLaneChangingAccelerationEnvironment"
    pass_params(env_name, sumo_params, sumo_binary, type_params, env_params, net_params,
                cfg_params, initial_config, scenario)

    #env = GymEnv("TwoIntersectionEnv-v0", force_reset=True, record_video=False)
    env = GymEnv(env_name+"-v0", record_video=False)
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
        batch_size=1500,
        max_path_length=horizon,
        n_itr=1000,  # 1000
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
        # python_command="/home/aboudy/anaconda2/envs/rllab3/bin/python3.5"
        # plot=True,
    )
