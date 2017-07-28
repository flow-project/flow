import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from cistar_dev.core.exp import SumoExperiment
from cistar_dev.envs.loop_velocity import SimpleVelocityEnvironment
from cistar_dev.scenarios.loop.loop_scenario import LoopScenario
# from cistar_dev.controllers.car_following_models import *

logging.basicConfig(level=logging.INFO)

tot_cars = 12
auton_cars = 12
human_cars = tot_cars - auton_cars

sumo_params = {"port": 8873, "time_step":0.001}

sumo_binary = "sumo"

type_params = {"rl":(auton_cars, None, None, 0)}

env_params = {"target_velocity": 8}

net_params = {"length": 840, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"traffic/cistar_dev/leah/net/"}

cfg_params = {"start_time": 0, "end_time":3000, "cfg_path":"traffic/cistar_dev-dev/leah/cfg/"}


# initial_positions = [("top", 0), ("top", 70), ("top", 140), \
#                     ("left", 0), ("left", 70), ("left", 140), \
#                     ("bottom", 0), ("bottom", 70), ("bottom", 140), \
#                     ("right", 0), ("right", 70), ("right", 140)]


initial_config = {"shuffle": False}

scenario = LoopScenario("leah-test-exp", type_params, net_params, cfg_params)#, initial_config=initial_config)

exp = SumoExperiment(SimpleVelocityEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

print("experiment initialized")

env = normalize(exp.env)

stub(globals())

for seed in [1]: # [1, 5, 10, 73, 56]
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(16,)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=2000,
        max_path_length=400,
        # whole_paths=True,
        n_itr=1500,
        # discount=0.99,
        # step_size=0.01,
    )
    # algo.train()

    print("IN RL BUILD TESTER, BEFORE RUN EXPERIMENT LITE")
    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="ec2",
        exp_prefix="leah-test-exp"
        # plot=True,
    )

exp.env.terminate()
