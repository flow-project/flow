"""Trains a percentage of AVs to increase traveling speeds in a figure 8."""

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from flow.scenarios.figure_eight import Figure8Scenario
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.core.params import VehicleParams
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoCarFollowingParams
from rllab.envs.gym_env import GymEnv

HORIZON = 1500


def run_task(*_):
    """Implement the run_task method needed to run experiments with rllab."""
    sim_params = SumoParams(sim_step=0.1, render=True)

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
            decel=1.5,
        ),
        num_vehicles=1)
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
            decel=1.5,
        ),
        num_vehicles=13)

    additional_env_params = {
        "target_velocity": 20,
        "max_accel": 3,
        "max_decel": 3,
        "sort_vehicles": False
    }
    env_params = EnvParams(
        horizon=HORIZON, additional_params=additional_env_params)

    additional_net_params = {
        "radius_ring": 30,
        "lanes": 1,
        "speed_limit": 30,
        "resolution": 40
    }
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform")

    print("XXX name", exp_tag)
    scenario = Figure8Scenario(
        exp_tag,
        vehicles,
        net_params,
        initial_config=initial_config)

    env_name = "AccelEnv"
    pass_params = (env_name, sim_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(16, 16))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=15000,
        max_path_length=horizon,
        n_itr=500,
        # whole_paths=True,
        discount=0.999,
        # step_size=v["step_size"],
    )
    algo.train(),


exp_tag = "figure-eight-control"

for seed in [5]:  # , 20, 68]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Keeps the snapshot parameters for all iterations
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        # plot=True,
    )
