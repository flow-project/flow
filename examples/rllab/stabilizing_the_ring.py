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
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

# recurrent stuff
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import *
from flow.controllers.car_following_models import *
from flow.controllers.routing_controllers import *
from flow.core.vehicles import Vehicles
from flow.core.params import *
from rllab.envs.gym_env import GymEnv

import numpy as np
import sys


def run_task(v):
    logging.basicConfig(level=logging.INFO)

    sumo_params = SumoParams(sim_step=0.1, sumo_binary="sumo", seed=0)

    vehicles = Vehicles()
    vehicles.add(veh_id="rl",
                 acceleration_controller=(RLController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=1)
    vehicles.add(veh_id="idm",
                 acceleration_controller=(IDMController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=21)

    additional_env_params = {"target_velocity": 8, "num_steps": 3600,
                             "scenario_type": LoopScenario}
    env_params = EnvParams(max_decel=-1, max_accel=1,
                           additional_params=additional_env_params)

    additional_net_params = {"length": 260, "lanes": 1, "speed_limit": 30,
                             "resolution": 40}
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform", bunching=50)

    print("XXX name", exp_tag)
    scenario = LoopScenario(exp_tag, CircleGenerator, vehicles, net_params,
                            initial_config=initial_config)

    env_name = "WaveAttenuationPOEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianGRUPolicy(
        env_spec=env.spec,
        hidden_sizes=(5,)
    )

    # policy = GaussianMLPPolicy(
    #     env_spec=env.spec,
    #     hidden_sizes=(16, 16)
    #     # hidden_sizes=(100, 50, 25)
    # )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=72000,
        max_path_length=horizon,
        n_itr=250,
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
        n_parallel=2,
        # Keeps the snapshot parameters for all iterations
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        # plot=True,
    )
