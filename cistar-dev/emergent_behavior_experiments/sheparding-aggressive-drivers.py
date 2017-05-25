"""
Script used for running shepherding aggressive drivers.

Two types of scenarios are implemented
 1. Aggressive drivers are also rl cars, in which case the reward function is modified to allow
    aggressive drivers to seek higher target velocities
 2. Aggressive drivers are human drivers, in which case their car-following characteristics are
    modified to demand higher velocities and accelerations

    TODO: also add aggressive nature in lane-changing behavior
"""

import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from cistar.envs.lane_changing import ShepherdAggressiveDrivers
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.rlcontroller import RLController
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.car_following_models import *

logging.basicConfig(level=logging.INFO)

stub(globals())

sumo_params = {"port": 8873, "time_step": 0.01}
sumo_binary = "sumo-gui"

test_type = 'rl'    # type of test being implemented (see comment at start of file)
num_aggressive = 1  # number of aggressive drivers
num_cars = 22       # total number of cars in simulation
num_human = 0       # number of uncontrollable (human) vehicles
num_auto = 22       # number of controllable (rl) vehicles
ind_aggressive = [0, 2, 4]  # index of aggressive cars

if test_type == 'rl':
    num_human = 0
    num_auto = num_cars
elif test_type == 'human_car_following':
    num_human = num_aggressive
    num_auto = num_cars - num_aggressive

exp_tag = str(num_cars) + 'car-shepherd'

type_params = {"rl": (num_auto, (RLController, {}), (StaticLaneChanger, {}), 0),
               "ovm": (num_human, (OVMController, {}), (StaticLaneChanger, {}), 0)}

env_params = {"target_velocity": 8, "target_velocity_aggressive": 12, "ind_aggressive": ind_aggressive,
              "max-deacc": -3, "max-acc": 3, "lane_change_duration": 5, "fail-safe": "None"}

net_params = {"length": 200, "lanes": 2, "speed_limit": 0, "resolution": 40, "net_path": "debug/net/"}

cfg_params = {"start_time": 0, "end_time": 30000, "cfg_path": "debug/cfg/"}

scenario = LoopScenario("two-lane-two-controller", type_params, net_params, cfg_params)

env = ShepherdAggressiveDrivers(env_params, sumo_binary, sumo_params, scenario)

env = normalize(env)

for seed in [5]:  # [5, 10, 73, 56, 1]
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(150, 25)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=30000,  # 4000
        max_path_length=1500,
        n_itr=1000,  # 50000
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
