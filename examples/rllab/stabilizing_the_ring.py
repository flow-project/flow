"""Train a single AV to stabilize a variable density ring road."""

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

from flow.scenarios.loop import LoopScenario
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from rllab.envs.gym_env import GymEnv

HORIZON = 3000


def run_task(*_):
    """Implement the run_task method needed to run experiments with rllab."""
    sim_params = SumoParams(sim_step=0.1, render=False, seed=0)

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1)
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=0,
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=21)

    additional_env_params = {
        "ring_length": [220, 270],
        "max_accel": 1,
        "max_decel": 1
    }
    env_params = EnvParams(
        horizon=HORIZON,
        additional_params=additional_env_params,
        warmup_steps=750,
        clip_actions=False
    )

    additional_net_params = {
        "length": 260,
        "lanes": 1,
        "speed_limit": 30,
        "resolution": 40
    }
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform", bunching=50)

    print("XXX name", exp_tag)
    scenario = LoopScenario(
        exp_tag,
        vehicles,
        net_params,
        initial_config=initial_config)

    env_name = "WaveAttenuationPOEnv"
    pass_params = (env_name, sim_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianGRUPolicy(
        env_spec=env.spec,
        hidden_sizes=(5, ),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=60000,
        max_path_length=horizon,
        n_itr=500,
        # whole_paths=True,
        discount=0.999,
        # step_size=v["step_size"],
    )
    algo.train(),


exp_tag = "stabilizing-the-ring"

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
