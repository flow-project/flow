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

from cistar_dev.envs.braess_paradox import BraessParadoxEnvironment
from cistar_dev.scenarios.braess_paradox.braess_paradox_scenario import BraessParadoxScenario
from cistar_dev.controllers.rlcontroller import RLController
from cistar_dev.controllers.lane_change_controllers import *
from cistar_dev.controllers.car_following_models import *

logging.basicConfig(level=logging.INFO)

stub(globals())

sumo_params = {"time_step": 0.1, "starting_position_shuffle": False, "vehicle_arrangement_shuffle": False,
               "rl_lc": "no_lat_collide", "human_lc": "no_lat_collide",
               "rl_sm": "no_collide", "human_sm": "no_collide"}
sumo_binary = "sumo-gui"

env_params = {"target_velocity": 4, "max-deacc": -6, "max-acc": 3, "fail-safe": "None",
              "observation_pos_std": 0, "observation_vel_std": 0, "human_acc_std": 0.5, "rl_acc_std": 0}

net_params = {"edge_length": 120, "angle": np.pi/9, "lanes": 1, "speed_limit": 30, "resolution": 40,
              "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/rl/cfg/"}

initial_config = {"shuffle": False}

num_cars = 22
num_auto = 1

exp_tag = str(num_cars) + "-car-" + str(num_auto) + "-rl-braess-paradox"

type_params = {"rl": (num_auto, (RLController, {}), (StaticLaneChanger, {}), 0),
               "idm": (num_cars - num_auto, (IDMController, {}), (StaticLaneChanger, {}), 0)}

scenario = BraessParadoxScenario(exp_tag, type_params, net_params, cfg_params, initial_config=initial_config)

env = BraessParadoxEnvironment(env_params, sumo_binary, sumo_params, scenario)

env = normalize(env)

for seed in [5]:  # [16, 20, 21, 22]:
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=60000,
        max_path_length=6000,
        n_itr=1000,
        # whole_paths=True,
        discount=0.999,
        step_size=0.01,
    )

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        python_command="/home/aboudy/anaconda2/envs/rllab3/bin/python3.5"
        # plot=True,
    )
