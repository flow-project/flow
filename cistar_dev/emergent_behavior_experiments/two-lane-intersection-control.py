"""
This script presents the use of two-way intersections in CISTAR.

Cars enter from the bottom and left nodes following a probability distribution, and
continue to move straight until they exit through the top and right nodes, respectively.
"""

from cistar.envs.intersection import SimpleIntersectionEnvironment
from cistar.envs.two_intersection import TwoIntersectionEnvironment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.intersections.intersection_scenario import *
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.rlcontroller import RLController

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

import pdb

def run_task(*_):
    env = GymEnv("TwoIntersectionEnv-v0")
    env = normalize(env)

    for seed in [5, 10]:  # [16, 20, 21, 22]:
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
            max_path_length=500,
            n_itr=1000,
            # whole_paths=True,
            # discount=0.999,
            step_size=0.01,
        )
        algo.train()
        

run_experiment_lite(
            run_task,
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="all",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=1,
            mode="local",
            exp_prefix='test-test',
            #python_command="/home/aboudy/anaconda2/envs/rllab3/bin/python3.5"
            # plot=True,
)

