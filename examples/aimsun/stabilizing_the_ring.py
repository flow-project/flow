"""Train a single AV to stabilize a variable density ring road."""

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from flow.scenarios.loop import LoopScenario
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.core.params import VehicleParams
from flow.core.params import AimsunParams, EnvParams, NetParams, InitialConfig
from rllab.envs.gym_env import GymEnv

HORIZON = 1500


def run_task(*_):
    """Implement the run_task method needed to run experiments with rllab."""
    sim_params = AimsunParams(sim_step=0.5, render=False, seed=0)

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1)
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=21)

    additional_env_params = {
        "target_velocity": 8,
        "ring_length": None,
        "max_accel": 1,
        "max_decel": 1
    }
    env_params = EnvParams(
        horizon=HORIZON,
        additional_params=additional_env_params,
        warmup_steps=1500)

    additional_net_params = {
        "length": 230,
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
    simulator = 'aimsun'
    pass_params = (env_name, sim_params, vehicles, env_params, net_params,
                   initial_config, scenario, simulator)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(3, 3),
    )

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
