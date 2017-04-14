import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from cistar.core.exp import SumoExperiment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.rlcontroller import RLController
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *
logging.basicConfig(level=logging.INFO)

tot_cars = 8

auton_cars = 4
human_cars = tot_cars - auton_cars

sumo_params = {"port": 8873, "time_step":0.1}

sumo_binary = "sumo-gui"

type_params = {"rl":(auton_cars, (RLController, {}), (StaticLaneChanger, {}), 0),
               "cfm":(human_cars, (BCMController, {"v_des":10}), (StaticLaneChanger, {}), 0)}

env_params = {"target_velocity": 8, "max-deacc":3, "max-acc":3}

net_params = {"length": 200, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"debug/rl/net/"}

cfg_params = {"start_time": 0, "end_time":3000, "cfg_path":"debug/rl/cfg/"}


# initial_positions = [("top", 0), ("top", 20), ("top", 40), \
#                     ("left", 0)]

initial_config = {"shuffle": False}#, "positions":initial_positions}

scenario = LoopScenario("rl-test", type_params, net_params, cfg_params, initial_config=initial_config)

exp = SumoExperiment(SimpleAccelerationEnvironment, env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

print("experiment initialized")

env = normalize(exp.env)

for seed in [1]: # [1, 5, 10, 73, 56]
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32,32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=200,
        max_path_length=1000,
        # whole_paths=True,
        n_itr=150,
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
        exp_prefix="leah-test-exp"
        # plot=True,
    )

exp.env.terminate()
