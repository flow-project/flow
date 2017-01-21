from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite

from SumoExperiment import SumoExperiment
from LoopExperiment import SimpleVelocityEnvironment

#from CircleGenerator import CircleGenerator


import subprocess
import sys


import logging

logging.basicConfig(level=logging.INFO)

sumo_params = {"port": 8873}

sumo_binary = "/Users/kanaad/code/research/learning-traffic/sumo/sumo-svn/bin/sumo-gui"

vehicle_controllers = {"rl": (2, None)}

env_params = {"target_velocity": 25}

initial_config = {}

net_params = {"length": 200, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"debug/net/"}

##data path needs to be relative to cfg location
cfg_params = {"type_list": ["rl"], "start_time": 0, "end_time":3000, "cfg_path":"debug/cfg/"}

#exp = SumoExperiment("test-exp", SumoEnvironment, env_params, 1, vehicle_controllers, sumo_binary, sumo_params, initial_config, CircleGenerator, net_params, cfg_params)

leah_sumo_params = {"port": 8873, "cfg":"/Users/kanaad/code/research/learning-traffic/sumo/learning-traffic/cistar/debug/cfg/test-exp-200m1l.sumo.cfg"}

exp = SumoExperiment("test-exp", SimpleVelocityEnvironment, env_params, 2, vehicle_controllers, sumo_binary, leah_sumo_params, initial_config) #, CircleGenerator, net_params, cfg_params)

print("experiment initialized")

stub(globals())

env = normalize(exp.env)

for seed in [1, 5, 10, 73, 56]:
# for _ in range(5):
    # seed = 1
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(16,)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=1300,
        max_path_length=130,
        # whole_paths=True,
        n_itr=90,
        # discount=0.99,
        # step_size=0.01,
    )
    print("begin training")

    # algo.train()
    #
    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        exp_prefix="traffic_27_1"
        # exp_name="traffic/23/1/" + str(seed)
        # plot=True,
    )