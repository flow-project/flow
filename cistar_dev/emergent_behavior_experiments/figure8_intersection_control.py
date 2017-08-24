"""
Script used to train vehicles to stop crashing longitudinally and on intersections.
"""

import logging
from collections import OrderedDict

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

# from cistar_dev.core.exp import SumoExperiment
from cistar_dev.envs.loop_accel import SimpleAccelerationEnvironment
from cistar_dev.scenarios.loop.loop_scenario import LoopScenario
from cistar_dev.scenarios.figure8.figure8_scenario import Figure8Scenario
from cistar_dev.controllers.car_following_models import *
from cistar_dev.controllers.rlcontroller import RLController
from cistar_dev.controllers.lane_change_controllers import *

num_cars = 14
num_auto = 14
exp_tag = str(num_cars) + '-car-' + str(num_auto) + '-rl-intersection-control'


def run_task(*_):
    # import cistar_dev.envs as cistar_envs
    logging.basicConfig(level=logging.INFO)

    sumo_params = {"time_step": 0.1, "shuffle": True,
                   "rl_lc": "no_collide", "human_lc": "no_lat_collide",
                   "rl_sm": "no_collide", "human_sm": "no_lat_collide"}
    sumo_binary = "sumo"

    env_params = {"target_velocity": 30, "max-deacc": -6, "max-acc": 3, "num_steps": 1500,
                  "observation_vel_std": 0, "observation_pos_std": 0, "human_acc_std": 0, "rl_acc_std": 0}

    net_params = {"radius_ring": 30, "lanes": 1, "speed_limit": 30, "resolution": 40,
                  "net_path": "debug/net/", "no-internal-links": False}

    cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/rl/cfg/"}

    initial_config = {"shuffle": False}

    num_cars = 14
    num_auto = 14
    type_params = [("rl", num_auto, (RLController, {}), (StaticLaneChanger, {}), 0),
                   ("idm", num_cars - num_auto, (IDMController, {}), (StaticLaneChanger, {}), 0)]

    exp_type = 0

    if exp_type == 1:
        num_cars = 14
        num_auto = 1
        # type_params = \
        #     OrderedDict([("rl", (1, (RLController, {}), (StaticLaneChanger, {}), 0)),
        #                  ("idm", (13, (IDMController, {}), (StaticLaneChanger, {}), 0))])
        type_params = {"rl": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm": (13, (IDMController, {}), (StaticLaneChanger, {}), 0)}

    elif exp_type == 2:
        num_cars = 14
        num_auto = 2
        type_params = {"rl": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm": (6, (IDMController, {}), (StaticLaneChanger, {}), 0),
                       "rl2": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm2": (6, (IDMController, {}), (StaticLaneChanger, {}), 0)}

    elif exp_type == 3:
        num_cars = 14
        num_auto = 4
        type_params = {"rl": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm": (3, (IDMController, {}), (StaticLaneChanger, {}), 0),
                       "rl2": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm2": (2, (IDMController, {}), (StaticLaneChanger, {}), 0),
                       "rl3": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm3": (3, (IDMController, {}), (StaticLaneChanger, {}), 0),
                       "rl4": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm4": (2, (IDMController, {}), (StaticLaneChanger, {}), 0)}

    elif exp_type == 4:
        num_cars = 14
        num_auto = 7
        type_params = {"rl": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm": (1, (IDMController, {}), (StaticLaneChanger, {}), 0),
                       "rl2": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm2": (1, (IDMController, {}), (StaticLaneChanger, {}), 0),
                       "rl3": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm3": (1, (IDMController, {}), (StaticLaneChanger, {}), 0),
                       "rl4": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm4": (1, (IDMController, {}), (StaticLaneChanger, {}), 0),
                       "rl5": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm5": (1, (IDMController, {}), (StaticLaneChanger, {}), 0),
                       "rl6": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm6": (1, (IDMController, {}), (StaticLaneChanger, {}), 0),
                       "rl7": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
                       "idm7": (1, (IDMController, {}), (StaticLaneChanger, {}), 0)}

    scenario = Figure8Scenario(exp_tag, type_params, net_params, cfg_params, initial_config=initial_config)

    from cistar_dev import pass_params
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
        batch_size=15000,
        max_path_length=horizon,
        n_itr=1000,
        # whole_paths=True,
        discount=0.999,
        step_size=0.01,
    )
    algo.train(),

for seed in [5]:  # , 20, 68]:  # , 100, 128]:
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
