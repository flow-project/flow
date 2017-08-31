"""
Script used to train test platooning on a single lane.

RL vehicles are bunched together. The emergent behavior we are hoping to witness
is that rl-vehicles group together in other to allow non rl-vehicles a larger headway,
and thus larger equilibrium speeds.

One concern is whether rl-vehicles will start tail-gating human vehicles.

Build is using tensorflow which appears to be broken.
"""

import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.algos.multi_trpo import MultiTRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.new_analogy.tf.policies.auto_mlp_policy import AutoMLPPolicy
from sandbox.rocky.tf.policies.multi_mlp_policy import MultiMLPPolicy
from sandbox.rocky.neural_learner.sample_processors.multi_sample_processor import MultiSampleProcessor


# from cistar.core.exp import SumoExperiment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.rlcontroller import RLController
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.car_following_models import *
from rllab.envs.gym_env import GymEnv
import sys

def run_task(v):
    import cistar.envs as cistar_envs
    logging.basicConfig(level=logging.INFO)

    sumo_params = {"time_step": 0.1, "rl_sm": "aggressive", "human_sm": "no_collide"}
    sumo_binary = "sumo"

    env_params = {"target_velocity": 15, "max-deacc": -6, "max-acc": 3, "fail-safe": "None",
                "num_steps": 1, "shared_policy": 1}

    net_params = {"length": 230, "lanes": 1, "speed_limit": 30, "resolution": 40,
                  "net_path": "debug/net/"}

    cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/rl/cfg/"}

    initial_config = {"shuffle": True}

    type_params = [("rl", 2, (RLController, {}), (StaticLaneChanger, {}), 0),
                   ("idm", 1, (IDMController, {}), (StaticLaneChanger, {}), 0)]

    scenario = LoopScenario(exp_tag, type_params, net_params, cfg_params, initial_config=initial_config)

    from cistar import pass_params
    env_name = "SimpleMultiAgentAccelerationEnvironment"
    pass_params = (env_name, sumo_params, sumo_binary, type_params, env_params, net_params,
                cfg_params, initial_config, scenario)

    main_env = GymEnv(env_name, record_video=False, register_params = pass_params)
    horizon = main_env.horizon
    # replace raw envs with wrapped shadow envs
    main_env._shadow_envs = [TfEnv(normalize(env)) for env in main_env.shadow_envs]
    sub_policies = [AutoMLPPolicy(
        name="sub-policy-%s" % i,
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
        batch_size=3,
        max_path_length=horizon,
        n_itr=1,
        #discount=0.99,
        step_size=v["step_size"],
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
        sample_processor_cls=MultiSampleProcessor,
        n_vectorized_envs=2,
    )
    algo.train()


exp_tag = str(1) + "-car-stabilizing-the-ring-multiagent-tf"
for step_size in [.01]:
    for seed in [5]:
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
            variant=dict(step_size=step_size, seed=seed)
            # python_command="/home/aboudy/anaconda2/envs/rllab3/bin/python3.5"
            # plot=True,
        )
