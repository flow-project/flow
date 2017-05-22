"""
Script used to test the incorporation of lane changing into rl vehicles.


"""

import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from cistar.envs.lane_changing import SimpleLaneChangingAccelerationEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.rlcontroller import RLController

logging.basicConfig(level=logging.INFO)

stub(globals())

sumo_params = {"port": 8873, "time_step": 0.1}
sumo_binary = "sumo-gui"
num_cars = 22

exp_tag = str(num_cars) + '-car-rl-lane_change'

type_params = {"rl": (num_cars, (RLController, {}), None, 0)}

env_params = {"target_velocity": 8, "max-deacc": -3, "max-acc": 3, "lane_change_duration": 5,
              "fail-safe": "None"}

net_params = {"length": 200, "lanes": 2, "speed_limit": 0, "resolution": 40, "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "end_time": 3000, "cfg_path": "debug/cfg/"}

scenario = LoopScenario("two-lane-two-controller", type_params, net_params, cfg_params)

env = SimpleLaneChangingAccelerationEnvironment(env_params, sumo_binary, sumo_params, scenario)

env = normalize(env)

for seed in [5]:  # [5, 10, 73, 56, 1]: # [1, 5, 10, 73, 56]
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(16,)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=8000,
        max_path_length=1000,
        n_itr=50000,  # 1000
        # whole_paths=True,
        # discount=0.99,
        # step_size=0.01,
    )
    # algo.train()

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        python_command="/home/aboudy/anaconda2/envs/rllab3/bin/python3.5"
        # plot=True,
    )
