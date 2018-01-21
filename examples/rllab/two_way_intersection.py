"""
Basic test of fully rl intersection environment with accelerations as actions.
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
from flow.envs.two_intersection import TwoIntersectionEnvironment
import flow.core.config as flow_config

from flow.scenarios.intersections.gen import TwoWayIntersectionGenerator
from flow.scenarios.intersections.intersection_scenario import TwoWayIntersectionScenario
from flow.controllers.rlcarfollowingcontroller import RLCarFollowingController

logging.basicConfig(level=logging.INFO)


def run_task(*_):
    auton_cars = 20

    sumo_params = SumoParams(sim_step=0.1,
                             sumo_binary="sumo-gui")

    vehicles = Vehicles()
    vehicles.add("idm", (RLCarFollowingController, {}), None, None, 0, 20)

    intensity = .2
    v_enter = 10
    env_params = EnvParams(additional_params={"target_velocity": v_enter,
                                              "control-length": 150, "max_speed": v_enter})

    additional_net_params = {"horizontal_length_in": 400, "horizontal_length_out": 800, "horizontal_lanes": 1,
                             "vertical_length_in": 400, "vertical_length_out": 800, "vertical_lanes": 1,
                             "speed_limit": {"horizontal": v_enter, "vertical": v_enter}}
    net_params = NetParams(no_internal_links=False, additional_params=additional_net_params)

    cfg_params = {"start_time": 0, "end_time": 3000, "cfg_path": "debug/cfg/"}

    initial_config = InitialConfig(spacing="custom", additional_params={"intensity": intensity, "enter_speed": v_enter})

    scenario = TwoWayIntersectionScenario("two-way-intersection", TwoWayIntersectionGenerator,
                                          vehicles, net_params, initial_config=initial_config)

    env = TwoIntersectionEnvironment(env_params, sumo_params, scenario)
    env_name = "TwoIntersectionEnv"
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
        hidden_sizes=(64, 64)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=30000,
        max_path_length=horizon,
        # whole_paths=True,
        n_itr=200,
        discount=0.999,
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
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="local",
        exp_prefix="intersection-exp",
        python_command=flow_config.PYTHON_COMMAND
        # plot=True,
    )
