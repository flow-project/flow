"""
Script used to train test platooning on a single lane.

RL vehicles are bunched together. The emergent behavior we are hoping to witness
is that rl-vehicles group together in other to allow non rl-vehicles a larger headway,
and thus larger equilibrium speeds.

One concern is whether rl-vehicles will start tail-gating human vehicles.
"""

import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.shared_trpo import SharedTRPO
from rllab.envs.proxy_env import ProxyEnv
from rllab.policies.shared_mlp_policy import SharedMLPPolicy
from sandbox.rocky.neural_learner.sample_processors.shared_sample_processor import SharedSampleProcessor
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


from cistar_dev.scenarios.loop.gen import CircleGenerator
from cistar_dev.scenarios.loop.loop_scenario import LoopScenario
from cistar_dev.controllers.rlcontroller import RLController
from cistar_dev.controllers.lane_change_controllers import *
from cistar_dev.controllers.car_following_models import *
from rllab.envs.gym_env import GymEnv

def run_task(*_):
    import cistar_dev.envs as cistar_envs
    logging.basicConfig(level=logging.INFO)

    sumo_params = {"time_step": 0.1, "rl_sm": "aggressive", "human_sm": "no_collide"}
    sumo_binary = "sumo"

    env_params = {"target_velocity": 15, "max-deacc": -6, "max-acc": 3, "fail-safe": "None",
                  "num_steps": 500, "shared_policy": 1}

    net_params = {"length": 230, "lanes": 1, "speed_limit": 30, "resolution": 40,
                  "net_path": "debug/net/"}

    cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/rl/cfg/"}

    initial_config = {"shuffle": True}

    type_params = [("rl", 2, (RLController, {}), (StaticLaneChanger, {}), 0),
                   ("idm", 1, (IDMController, {}), (StaticLaneChanger, {}), 0)]

    scenario = LoopScenario(exp_tag, CircleGenerator, type_params, net_params,
                            cfg_params=cfg_params, initial_config=initial_config)

    env_name = "SimpleMultiAgentAccelerationEnvironment"
    pass_params = (env_name, sumo_params, sumo_binary, type_params, env_params, net_params,
                   cfg_params, initial_config, scenario)

    main_env = GymEnv(env_name, record_video=False, register_params = pass_params, force_reset=True)
    horizon = main_env.horizon
    # replace raw envs with wrapped shadow envs
    main_env._shadow_envs = [ProxyEnv(normalize(env)) for env in main_env.shadow_envs]
    sub_policy = [GaussianMLPPolicy(
        # TOFIX(eugene): what does env_spec do, and is this a problem right hurr
        env_spec=main_env._shadow_envs[0].spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )]

    policy = SharedMLPPolicy(
        name="policy",
        env_spec=[env.spec for env in main_env.shadow_envs],
        policies=sub_policy
    )
    baseline = LinearFeatureBaseline(env_spec=main_env.spec)

    algo = SharedTRPO(
        env=main_env,
        policy=policy,
        baseline=baseline,
        batch_size=30000,
        max_path_length=horizon,
        n_itr=200,
        #discount=0.99,
        step_size=.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
        sample_processor_cls=SharedSampleProcessor,
        # n_vectorized_envs=1,
    )
    algo.train()


exp_tag = str(22) + "-car-stabilizing-the-ring-multiagent"
for seed in [5, 10, 15]:
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
        #python_command="/home/aboudy/anaconda2/envs/rllab3/bin/python3.5"
        # plot=True,
    )
