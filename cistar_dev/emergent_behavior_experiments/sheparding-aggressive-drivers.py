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
#from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv

from cistar.envs.lane_changing import *
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.rlcontroller import RLController
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.car_following_models import *

num_cars = 22  # number of uncontrollable (human) vehicles
num_drunk = 1  # number of drunk drivers in the ring
num_auto = 2   # number of controllable (rl) vehicles
# parameters describing drunk drivers
perturb_time = 5
perturb_size = 40
exp_tag = str(num_cars - num_auto - num_drunk) + '-human-' + str(num_drunk) + '-drunk-' + str(num_auto) + \
    '-rl-shepherding-aggressive-drivers-' + str(perturb_time) + '-perturb-time-' + str(perturb_size) + '-perturb-size'


def run_task(*_):
    import cistar.envs as cistar_envs
    logging.basicConfig(level=logging.INFO)

    sumo_params = {"time_step": 0.1,
                   "rl_lc": "no_lat_collide", "human_lc": "strategic",
                   "human_sm": "no_collide", "rl_sm": "no_collide"}

    sumo_binary = "sumo"

    # all human cars constrained to right lane
    type_params = [
        ("rl", num_auto, (RLController, {}), None, 0),
        ("idm", num_cars - num_auto - num_drunk, (IDMController, {}), (StaticLaneChanger, {}), 0),
        ("drunk", num_drunk, (DrunkDriver, {"perturb_time": perturb_time}), (StaticLaneChanger, {}), 0)]

    # human cars can lane change
    # type_params = {
    #     "rl": (num_auto, (RLController, {}), None, 0),
    #     "idm": (num_cars - num_auto - num_drunk, (IDMController, {}), None, 0),
    #     "drunk": (num_drunk, (DrunkDriver, {"perturb_time": perturb_time, "perturb_size": perturb_size}), None, 0)}

    env_params = {"target_velocity": 8, "max-deacc": -6, "max-acc": 3,
                  "lane_change_duration": 3, "fail-safe": "None", "num_steps":300}

    net_params = {"length": 230, "lanes": 2, "speed_limit": 30, "resolution": 40, "net_path": "debug/net/"}

    cfg_params = {"start_time": 0, "end_time": 30000000, "cfg_path": "debug/cfg/"}

    initial_config = {"shuffle": True}

    scenario = LoopScenario("two-lane-two-controller", type_params, net_params, cfg_params, initial_config=initial_config)

    from cistar import pass_params
    env_name = "RLOnlyLane"
    pass_params = (env_name, sumo_params, sumo_binary, type_params, env_params, net_params,
                cfg_params, initial_config, scenario)

    #env = GymEnv("TwoIntersectionEnv-v0", force_reset=True, record_video=False)
    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # hidden_sizes=(200, 100, 50)
        hidden_sizes=(100, 50, 25)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=1500,  # 4000
        max_path_length=horizon,
        n_itr=2,  # 50000

        # whole_paths=True,
        #discount=0.99,
        step_size=0.01,
    )
    algo.train(),


for seed in [5]:  # [5, 10, 73, 56, 1]: # [1, 5, 10, 73, 56]
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=4,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        # python_command="/home/aboudy/anaconda2/envs/rllab3/bin/python3.5"
        # plot=True,
    )
