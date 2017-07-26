"""
"""

import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

# from cistar.core.exp import SumoExperiment
from cistar.envs.lane_changing import LaneChangeOnlyEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.rlcontroller import RLController
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.car_following_models import *

logging.basicConfig(level=logging.INFO)

stub(globals())

sumo_params = {"time_step": 0.1, "starting_position_shuffle": False, "vehicle_arrangement_shuffle": False,
               "rl_lc": "no_lat_collide", "human_lc": "no_lat_collide",
               "rl_sm": "no_collide", "human_sm": "no_collide"}
sumo_binary = "sumo-gui"

env_params = {"target_velocity": 8, "rl_acc_controller": IDMController, "lane_change_duration": 0,
              "observation_pos_std": 0, "observation_vel_std": 0, "human_acc_std": 0.5, "rl_acc_std": 0.5}

net_params = {"length": 230, "lanes": 2, "speed_limit": 30, "resolution": 40,
              "net_path": "debug/net/", "lanes_distribution": 2}

cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/rl/cfg/"}

initial_config = {"shuffle": False, "spacing": "gaussian", "downscale": 10}

num_cars = 44
num_auto = 1

exp_tag = str(num_cars) + "-car-" + str(num_auto) + "-lane-change-only-control"

type_params = {"rl": (num_auto, (RLController, {}), None, 0),
               "idm": (num_cars - num_auto, (IDMController, {}), None, 0)}

scenario = LoopScenario(exp_tag, type_params, net_params, cfg_params, initial_config=initial_config)

env = LaneChangeOnlyEnvironment(env_params, sumo_binary, sumo_params, scenario)

env = normalize(env)

for seed in [5]:  # [16, 20, 21, 22]:
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=20000,
        max_path_length=2000,
        n_itr=1000,
        # whole_paths=True,
        discount=0.999,
        step_size=0.01,
    )

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        python_command="/home/aboudy/anaconda2/envs/rllab3/bin/python3.5"
        # plot=True,
    )
