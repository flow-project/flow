"""Trains cooperative merging behavior in an open merge network.

Trains a small percentage of rl vehicles to dissipate shockwaves caused by
merges in an open network.
"""

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from flow.scenarios.merge import MergeScenario, ADDITIONAL_NET_PARAMS
from flow.controllers import RLController, IDMController
from flow.core.params import VehicleParams
from flow.core.params import SumoParams, InFlows, EnvParams, NetParams, \
    InitialConfig, SumoCarFollowingParams

# experiment number
# - 0: 10% RL penetration,  5 max controllable vehicles
# - 1: 25% RL penetration, 13 max controllable vehicles
# - 2: 33% RL penetration, 17 max controllable vehicles
EXP_NUM = 0

# time horizon of a single rollout
HORIZON = 600
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 1

# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = [0.1, 0.25, 0.33][EXP_NUM]
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = [5, 13, 17][EXP_NUM]


def run_task(_):
    """Implement the run_task method needed to run experiments with rllab."""
    sim_params = SumoParams(
        render=True, sim_step=0.2, restart_instance=True)

    # RL vehicles constitute 5% of the total number of vehicles
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        num_vehicles=5)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        num_vehicles=0)

    # Vehicles are introduced from both sides of merge, with RL vehicles
    # entering from the highway portion as well
    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="inflow_highway",
        vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="rl",
        edge="inflow_highway",
        vehs_per_hour=RL_PENETRATION * FLOW_RATE,
        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="inflow_merge",
        vehs_per_hour=100,
        departLane="free",
        departSpeed=7.5)

    additional_env_params = {
        "target_velocity": 25,
        "num_rl": NUM_RL,
        "max_accel": 1.5,
        "max_decel": 1.5
    }
    env_params = EnvParams(
        horizon=HORIZON,
        sims_per_step=5,
        warmup_steps=0,
        additional_params=additional_env_params)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["merge_lanes"] = 1
    additional_net_params["highway_lanes"] = 1
    additional_net_params["pre_merge_length"] = 500
    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing="uniform", lanes_distribution=float("inf"))

    scenario = MergeScenario(
        name="merge-rl",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env_name = "WaveAttenuationMergePOEnv"
    pass_params = (env_name, sim_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32, 32),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=HORIZON * N_ROLLOUTS,
        max_path_length=HORIZON,
        n_itr=1000,
        # whole_paths=True,
        discount=0.999,
    )
    algo.train(),


exp_tag = "stabilizing_highway_%.3f" % RL_PENETRATION

for seed in [5]:  # , 20, 68, 72, 125]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=N_CPUS,
        # Keeps the snapshot parameters for all iterations
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        # plot=True,
        sync_s3_pkl=True,
    )
