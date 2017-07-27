import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

# from cistar_dev.core.exp import SumoExperiment
from cistar_dev.envs.loop_accel_emission import SimpleAccelerationEnvironment
from cistar_dev.scenarios.loop.loop_scenario import LoopScenario
from cistar_dev.controllers.rlcontroller import RLController
from cistar_dev.controllers.lane_change_controllers import StaticLaneChanger

logging.basicConfig(level=logging.INFO)

stub(globals())

tot_cars = 12

auton_cars = 12
human_cars = tot_cars - auton_cars

sumo_params = {"port": 8880, "time_step":0.01}

sumo_binary = "sumo"

type_params = {"rl":(auton_cars, (RLController, {}), (StaticLaneChanger, {}), 0)}

env_params = {"target_velocity": 25, "max-vel":35, "min-vel":0, "max-deacc": 6, "max-acc":5}

net_params = {"length": 840, "lanes": 1, "speed_limit":35, "resolution": 40, "net_path":"emission/rl/net/"}

cfg_params = {"start_time": 0, "end_time":3000, "cfg_path":"emission/rl/cfg/"}


# initial_positions = [("top", 0), ("top", 70), ("top", 140), \
#                     ("left", 0), ("left", 70), ("left", 140), \
#                     ("bottom", 0), ("bottom", 70), ("bottom", 140), \
#                     ("right", 0), ("right", 70), ("right", 140)]



initial_config = {"shuffle": False}

scenario = LoopScenario("rl-emission-accel", type_params, net_params, cfg_params)#, initial_config=initial_config)

env = SimpleAccelerationEnvironment(env_params, sumo_binary, sumo_params, scenario)

logging.info("Experiment Set Up complete")

print("experiment initialized")

env = normalize(env)

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
        batch_size=400,
        max_path_length=2000,
        # whole_paths=True,
        n_itr=10,
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
        exp_prefix="rl-emission-accel"
        # plot=True,
    )

# exp.env.terminate()
