"""
Script used to train test platooning on a single lane.

RL vehicles are bunched together. The emergent behavior we are hoping to witness
is that rl-vehicles group together in other to allow non rl-vehicles a larger headway,
and thus larger equilibrium speeds.

One concern is whether rl-vehicles will start trail-gating human vehicles.
"""

import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

# from cistar.core.exp import SumoExperiment
# from cistar.envs.loop_accel import SimpleAccelerationEnvironment, SimplePartiallyObservableEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.rlcontroller import RLController
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.car_following_models import *
from rllab.envs.gym_env import GymEnv
import sys


def run_task(v):
    import cistar.envs as cistar_envs
    logging.basicConfig(level=logging.INFO)

    sumo_params = {"time_step": 0.1,
                   "starting_position_shuffle": True, "vehicle_arrangement_shuffle": True, "bunching_range": [0, 100],
                   "rl_lc": "aggressive", "human_lc": "aggressive", "rl_sm": "aggressive", "human_sm": "aggressive"}
    sumo_binary = "sumo-gui"

    env_params = {"target_velocity": 8, "max-deacc": -6, "max-acc": 3, "fail-safe": "None", "num_steps": 1500,
                  "observation_pos_std": 0, "observation_vel_std": 0, "human_acc_std": 0, "rl_acc_std": 0}

    net_params = {"length": 230, "lanes": 1, "speed_limit": 30, "resolution": 40,
                  "net_path": "debug/net/"}

    cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/rl/cfg/"}

    initial_config = {"shuffle": False, "spacing": "gaussian", "downscale": 10}

    num_cars = 22

    type_params = [("rl", 1, (RLController, {}), (StaticLaneChanger, {}), 0),
                   ("idm", num_cars - 1, (IDMController, {}), (StaticLaneChanger, {}), 0)]

    # type_params = {"rl": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
    #                "idm": (num_cars - 1, (IDMController, {}), (StaticLaneChanger, {}), 0)}

    scenario = LoopScenario(exp_tag, type_params, net_params, cfg_params, initial_config=initial_config)

    env_name = "SimpleAccelerationEnvironment"
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
        n_itr=500,
        # whole_paths=True,
        # discount=0.999,
        # step_size=v["step_size"],
    )
    algo.train(),

exp_tag = str(22) + "-car-stabilizing-the-ring-local-robust-0-std"

for seed in [5]:  # , 20, 68]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Keeps the snapshot parameters for all iterations
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        python_command="/home/aboudy/anaconda2/envs/rllab-distributed/bin/python3.5"
        # plot=True,
    )
