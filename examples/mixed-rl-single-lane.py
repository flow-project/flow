''' Basic test of fully rl environment with accelerations as actions. Single lane. Mixed human and rl.  

Variables:
    sumo_params {dict} -- [Pass time step, whether safe mode is on or off]
    sumo_binary {str} -- [Use either sumo-gui or sumo for visual or non-visual]
    type_params {dict} -- [Types of cars in the system. 
    Format {"name": (number, (Model, {params}), (Lane Change Model, {params}), initial_speed)}]
    env_params {dict} -- [Params for reward function]
    net_params {dict} -- [Params for network.
                            length: road length
                            lanes
                            speed limit
                            resolution: number of edges comprising ring
                            net_path: where to store net]
    cfg_params {dict} -- [description]
    initial_config {dict} -- [shuffle: randomly reorder cars to start experiment
                                spacing: if gaussian, add noise in start positions
                                bunching: how close to place cars at experiment start]
    scenario {[type]} -- [Which road network to use]
'''
import logging

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.core import config as flow_config

from flow.controllers.rlcontroller import RLController
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.controllers.routing_controllers import *

from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.controllers.rlcontroller import RLController
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *

from flow.core.params import SumoParams
from flow.core.params import EnvParams

def run_task(*_):
    import flow.envs as flow_envs
    logging.basicConfig(level=logging.INFO)

    tot_cars = 8
    auton_cars = 4
    human_cars = tot_cars - auton_cars

    sumo_params = SumoParams(time_step=0.1, sumo_binary="sumo-gui")

    vehicles = Vehicles()
    vehicles.add_vehicles("rl", (RLController, {}), (StaticLaneChanger, {}), (ContinuousRouter, {}), 0, auton_cars)
    vehicles.add_vehicles("cfm", (CFMController, {}), (StaticLaneChanger, {}), (ContinuousRouter, {}), 0, human_cars)

    additional_env_params = {"target_velocity": 8, "num_steps": 1000}
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = {"length": 200, "lanes": 1, "speed_limit": 30, "resolution": 40}
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig()

    scenario = LoopScenario("rl-test", CircleGenerator, vehicles, net_params, initial_config)

    env_name = "SimpleAccelerationEnvironment"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    logging.info("Experiment Set Up complete")

    print("experiment initialized")

    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=1000,
        max_path_length=horizon,
        # whole_paths=True,
        n_itr=2,
        # discount=0.99,
        # step_size=0.01,
    )
    algo.train()

for seed in [1]:  # [1, 5, 10, 73, 56]
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="local",
        exp_prefix="leah-test-exp",
        python_command=flow_config.PYTHON_COMMAND
        # plot=True,
    )
