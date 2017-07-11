"""
Script used to train test platooning on a single lane.

RL vehicles are bunched together. The emergent behavior we are hoping to witness
is that rl-vehicles group together in other to allow non rl-vehicles a larger headway,
and thus larger equilibrium speeds.

One concern is whether rl-vehicles will start trail-gating human vehicles.
"""

import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

# from cistar.core.exp import SumoExperiment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.rlcontroller import RLController
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.car_following_models import *

logging.basicConfig(level=logging.INFO)

stub(globals())

sumo_params = {"time_step": 0.1, "traci_control": 1, "rl_sm": "aggressive", "human_sm": "aggressive"}
sumo_binary = "sumo"

env_params = {"target_velocity": 8, "max-deacc": -6, "max-acc": 3, "fail-safe": "None"}

net_params = {"length": 230, "lanes": 1, "speed_limit": 30, "resolution": 40,
              "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/rl/cfg/"}

initial_config = {"shuffle": False}

num_cars = 22

exp_tag = str(num_cars) + "-car-stabilizing-the-ring-with-perturbations"

type_params = {
    "rl": (1, (RLController, {}), (StaticLaneChanger, {}), 0),
    "drunk": (1, (DrunkDriver, {}), (StaticLaneChanger, {}), 0),
    "idm": (num_cars - 2, (IDMController, {}), (StaticLaneChanger, {}), 0)}

scenario = LoopScenario(exp_tag, type_params, net_params, cfg_params, initial_config=initial_config)

env = SimpleAccelerationEnvironment(env_params, sumo_binary, sumo_params, scenario)

env = normalize(env)

for seed in [5]:
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
<<<<<<< Updated upstream:cistar-dev/emergent_behavior_experiments/stabilizing-the-ring-with-perturbations.py
<<<<<<< Updated upstream:cistar-dev/emergent_behavior_experiments/stabilizing-the-ring-with-perturbations.py
        batch_size=15000,
        max_path_length=1500,
        n_itr=1000,  # 1000
=======
=======
>>>>>>> Stashed changes:cistar-dev/emergent_behavior_experiments/mixed-human-rl.py
        batch_size=30000,
        max_path_length=2000,
        n_itr=400,  # 1000
>>>>>>> Stashed changes:cistar-dev/emergent_behavior_experiments/mixed-human-rl.py
        # whole_paths=True,
        discount=0.999,
        step_size=0.01,
    )

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=8,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="ec2",
        exp_prefix=exp_tag,
        # python_command="/home/aboudy/anaconda2/envs/rllab3/bin/python3.5"
        # plot=True,
    )
