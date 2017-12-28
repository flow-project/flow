"""
Basic implementation of a mixed-rl multi-lane environment with accelerations and
lane-changes as actions for the autonomous vehicles.
"""
import logging
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite, stub
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.core import config as flow_config

from flow.controllers.routing_controllers import *

from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.controllers.rlcontroller import RLController
from flow.controllers.car_following_models import *

logging.basicConfig(level=logging.INFO)


def run_task(*_):
    tot_cars = 8
    auton_cars = 5
    human_cars = tot_cars - auton_cars

    sumo_params = SumoParams(sim_step=0.1, sumo_binary="sumo-gui")

    vehicles = Vehicles()
    vehicles.add_vehicles(veh_id="rl",
                          acceleration_controller=(RLController, {}),
                          routing_controller=(ContinuousRouter, {}),
                          num_vehicles=auton_cars)
    vehicles.add_vehicles(veh_id="human",
                          acceleration_controller=(IDMController, {}),
                          routing_controller=(ContinuousRouter, {}),
                          num_vehicles=human_cars)

    additional_env_params = {"target_velocity": 8, "num_steps": 500}
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = {"length": 200, "lanes": 2, "speed_limit": 35,
                             "resolution": 40}
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig()

    scenario = LoopScenario(name="rl-test",
                            generator_class=CircleGenerator,
                            vehicles=vehicles,
                            net_params=net_params,
                            initial_config=initial_config)

    env_name = "SimpleLaneChangingAccelerationEnvironment"
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
        batch_size=10000,
        max_path_length=horizon,
        # whole_paths=True,
        n_itr=2,
        # discount=0.99,
        # step_size=0.01,
    )
    algo.train()

for seed in [1]: # [1, 5, 10, 73, 56]
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode="local",
        exp_prefix="leah-test-exp",
        python_command=flow_config.PYTHON_COMMAND
        # plot=True,
    )
