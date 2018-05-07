"""
Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
merges in an open network.
"""

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from flow.scenarios.merge.gen import MergeGenerator
from flow.scenarios.merge.scenario import MergeScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.rlcontroller import RLController
from flow.controllers.car_following_models import IDMController
from flow.core.vehicles import Vehicles
from flow.core.params import SumoParams, InFlows, EnvParams, NetParams, \
    InitialConfig

HORIZON = 3600

# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = 0.05


def run_task(_):
    sumo_params = SumoParams(sumo_binary="sumo-gui",
                             sim_step=0.2,
                             restart_instance=True)

    # RL vehicles constitute 5% of the total number of vehicles
    vehicles = Vehicles()
    vehicles.add(veh_id="human",
                 acceleration_controller=(IDMController, {"noise": 0.2}),
                 num_vehicles=5)
    vehicles.add(veh_id="rl",
                 acceleration_controller=(RLController, {}),
                 num_vehicles=0)

    # Vehicles are introduced from both sides of merge, with RL vehicles
    # entering from the highway portion as well
    inflow = InFlows()
    inflow.add(veh_type="human", edge="inflow_highway",
               vehs_per_hour=(1-RL_PENETRATION) * FLOW_RATE,
               departLane="free", departSpeed=10)
    inflow.add(veh_type="rl", edge="inflow_highway",
               vehs_per_hour=RL_PENETRATION * FLOW_RATE,
               departLane="free", departSpeed=10)
    inflow.add(veh_type="human", edge="inflow_merge", vehs_per_hour=100,
               departLane="free", departSpeed=7.5)

    additional_env_params = {"target_velocity": 25, "num_rl": 5,
                             "max_accel": 1.5, "max_decel": 1.5}
    env_params = EnvParams(horizon=HORIZON,
                           sims_per_step=5,
                           warmup_steps=0,
                           additional_params=additional_env_params)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["merge_lanes"] = 1
    additional_net_params["highway_lanes"] = 1
    additional_net_params["pre_merge_length"] = 500
    net_params = NetParams(in_flows=inflow,
                           no_internal_links=False,
                           additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform",
                                   lanes_distribution=float("inf"))

    scenario = MergeScenario(name="merge-rl",
                             generator_class=MergeGenerator,
                             vehicles=vehicles,
                             net_params=net_params,
                             initial_config=initial_config)

    env_name = "WaveAttenuationMergePOEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
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
        batch_size=18000,
        max_path_length=horizon,
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
        n_parallel=1,
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
