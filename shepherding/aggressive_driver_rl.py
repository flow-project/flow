"""
Used to train a mixed environment with an rl aggressive driver and IDM
controlled cars. 1 Lane.
"""
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.controllers.routing_controllers import *
from flow.core.vehicles import Vehicles
from flow.core.params import *

from flow.controllers.rlcarfollowingcontroller import RLCarFollowingController
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO 
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline 
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy 
from rllab.envs.gym_env import GymEnv 

num_rl_cars = 1
num_human_cars = 30

def run_task(v):
    sumo_params = SumoParams(time_step=0.1, sumo_binary="sumo-gui",
                             starting_position_shuffle=True,
                             vehicle_arrangement_shuffle=True)

    additional_env_params = {"target_velocity": 15, "num_steps": 1000}
    env_params = EnvParams(additional_params=additional_env_params,
                           lane_change_duration=0.1)

    additional_net_params = {"length": 500, "lanes": 4, "speed_limit": 15,
                             "resolution": 40}
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform_in_lane", lanes_distribution=4)

    # Sumo car-following parameters for human-driven vehicles
    human_cfm_params = SumoCarFollowingParams(sigma=1.0, tau=3.0)
    human_lc_params = SumoLaneChangeParams(
        model="SL2015", lcKeepRight=0, lcAssertive=0.5,
        lcSpeedGain=1.5, lcSpeedGainRight=1.0)

    # Sumo car-following params for the rl vehicle. These are supposed to allow
    # the vehicle to move above the speed limit and accelerate/decelerate at
    # higher magnitudes than the human vehicles
    aggressive_cfm_params = \
        SumoCarFollowingParams(speedFactor=1.75, decel=7.5, accel=4.5, tau=0.2)

    vehicles = Vehicles()
    vehicles.add_vehicles(veh_id="rl",
                          acceleration_controller=(RLCarFollowingController, {}),
                          routing_controller=(ContinuousRouter, {}),
                          num_vehicles=num_rl_cars,
                          sumo_lc_params=human_lc_params,
                          sumo_car_following_params=aggressive_cfm_params,
                          lane_change_controller=(SumoLaneChangeController, {}))
    vehicles.add_vehicles(veh_id="human",
                          acceleration_controller=(IDMController, {}),
                          routing_controller=(ContinuousRouter, {}),
                          num_vehicles=num_human_cars,
                          sumo_car_following_params=human_cfm_params,
                          sumo_lc_params=human_lc_params,
                          lane_change_controller=(SumoLaneChangeController, {}))


    scenario = LoopScenario("aggressive-rl-vehicle", CircleGenerator,
                            vehicles, net_params, initial_config)

    # Environment is specified differently than in non-autonomous
    env_name = "AggressiveDriverEnvironment"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)
    env = GymEnv(env_name, record_video=False, register_params=pass_params)

    # Necessary rllab components
    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(16, 16, 16)
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=30000,
        max_path_length=env.horizon,
        n_itr=2000,
        discount=0.999
    )
    algo.train()


# Run experiment in rllab
for seed in [5, 10, 15, 20]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        # n_parallel=8,
        # Keeps snapshot parameters for all iterations
        snapshot_mode="all",
        # Specifies seed for the experiment. If this is not provided, a random
        # seed will be used
        seed=seed,
        # mode="",  # ec2  # local_docker
        exp_prefix="aggressive-rl-vehicle",
        python_command="/Users/kanaad/anaconda3/envs/flow/bin/python",

    )
