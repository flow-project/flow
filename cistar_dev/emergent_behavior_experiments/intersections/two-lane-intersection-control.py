"""
This script presents the use of two-way intersections in cistar_dev.

Cars enter from the bottom and left nodes following a probability distribution, and
continue to move straight until they exit through the top and right nodes, respectively.
"""

from cistar_dev.envs.intersection import SimpleIntersectionEnvironment
from cistar_dev.envs.two_intersection import TwoIntersectionEnvironment
from cistar_dev.envs.loop_accel import SimpleAccelerationEnvironment
from cistar_dev.scenarios.intersections.intersection_scenario import *
from cistar_dev.controllers.car_following_models import *
from cistar_dev.controllers.lane_change_controllers import *
from cistar_dev.controllers.rlcontroller import RLController

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

import pdb


def run_task(v):
    import cistar_dev.envs as cistar_envs
    logging.basicConfig(level=logging.INFO)

    sumo_params = {"time_step": 0.1, "emission_path": "./data/",
                   "starting_position_shuffle": 0, "rl_sm": "aggressive"}
    sumo_binary = "sumo-gui"

    num_cars = 15

    # type_params = {"idm": (1, (IDMController, {}), (StaticLaneChanger, {}), 0)}
    type_params = [("rl", num_cars, (RLController, {}), None, 0.0)]

    # 1/intensity is the average time-spacing of the cars
    intensity = .3
    v_enter = 20.0

    env_params = {"target_velocity": v_enter, "max-deacc": -6, "max-acc": 6,
                  "control-length": 150, "max_speed": v_enter, "num_steps": 500}

    net_params = {"horizontal_length_in": 600, "horizontal_length_out": 1000, "horizontal_lanes": 1,
                  "vertical_length_in": 600, "vertical_length_out": 1000, "vertical_lanes": 1,
                  "speed_limit": {"horizontal": 30, "vertical": 30},
                  "net_path": "debug/net/"}

    cfg_params = {"start_time": 0, "end_time": 3000000, "cfg_path": "debug/cfg/"}

    initial_config = {"spacing": "custom", "intensity": intensity, "enter_speed": v_enter}

    scenario = TwoWayIntersectionScenario("figure8", type_params, net_params, cfg_params, initial_config=initial_config)

    env_name = "TwoIntersectionEnvironment"
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
        batch_size=1000,
        max_path_length=horizon,
        n_itr=2,
        # whole_paths=True,
        # discount=0.999,
        step_size=v["step_size"],
    )
    algo.train()


for step_size in [0.01]:
    for seed in [3]:  # [16, 20, 21, 22]:
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
            exp_prefix='test-test',
            variant=dict(step_size=step_size, seed=seed),
            python_command="/home/aboudy/anaconda2/envs/rllab-distributed/bin/python3.5"
            # plot=True,
        )


