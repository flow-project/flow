"""
This script presents the use of two-way intersections in cistar_dev.

Cars enter from the bottom and left nodes following a probability distribution, and
continue to move straight until they exit through the top and right nodes, respectively.
"""


import logging

from cistar_dev.envs.two_intersection import TwoIntersectionEnvironment
from cistar_dev.envs.loop_accel import SimpleAccelerationEnvironment
from cistar_dev.scenarios.intersections.intersection_scenario import *
from cistar_dev.controllers.car_following_models import *
from cistar_dev.controllers.lane_change_controllers import *
from cistar_dev.controllers.rlcontroller import RLController

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.gym_env import GymEnv

from rllab.algos.multi_trpo import MultiTRPO
from rllab.envs.proxy_env import ProxyEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.multi_mlp_policy import MultiMLPPolicy
from sandbox.rocky.neural_learner.sample_processors.multi_sample_processor import MultiSampleProcessor

import pdb

def run_task(v):
    import cistar_dev.envs as cistar_envs

    logging.basicConfig(level=logging.INFO)

    sumo_params = {"time_step": 0.1, "emission_path": "./data/",
                   "starting_position_shuffle": 1, "rl_sm": "aggressive"}
    sumo_binary = "sumo"

    num_cars = 3

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

    env_name = 'TwoIntersectionMultiAgentEnvironment'

    pass_params = (env_name, sumo_params, sumo_binary, type_params, env_params, net_params,
                   cfg_params, initial_config, scenario)

    main_env = GymEnv(env_name, record_video=False, register_params = pass_params, force_reset=True)
    horizon = main_env.horizon
    # replace raw envs with wrapped shadow envs
    main_env._shadow_envs = [ProxyEnv(normalize(env)) for env in main_env.shadow_envs]
    sub_policies = [GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    ) for i,env in enumerate(main_env.shadow_envs)]

    policy = MultiMLPPolicy(
        name="policy",
        env_spec=[env.spec for env in main_env.shadow_envs],
        policies=sub_policies
    )
    baselines = [LinearFeatureBaseline(env_spec=env.spec) for env in main_env.shadow_envs]

    algo = MultiTRPO(
        env=main_env,
        policy=policy,
        baselines=baselines,
        batch_size=1000,
        max_path_length=horizon,
        n_itr=2,
        #discount=0.99,
        step_size=v["step_size"],
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
        sample_processor_cls=MultiSampleProcessor,
        n_vectorized_envs=2,
    )
    algo.train()

for step_size in [0.01]:
    for seed in [1]:
        run_experiment_lite(
            run_task,
            exp_prefix="first_exp",
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="all",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=seed,
            mode="local",
            # mode="local_docker",
            # mode="ec2",
            variant=dict(step_size=step_size, seed=seed),
            # plot=True,
            # terminate_machine=False,
        )

