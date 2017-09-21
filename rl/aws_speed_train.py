from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from traffic.speed_env import SpeedEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

# for ctrl in [1, 2, 3, 4, 6, 12]:
# for aws, add "/root/code/rllab" to the path
cfgfn = "highway_files/leah-3/leah-3.sumo.cfg"
# Try to minimize remainder total_vehicles % ctrl
ctrl = 12
env = normalize(SpeedEnv(ctrl, 12, cfgfn))
for seed in [1, 5, 10, 73, 56]:
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(16,)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=2000,
        max_path_length=400,
        # whole_paths=True,
        n_itr=1500,
        # discount=0.99,
        # step_size=0.01,
    )
    # algo.train()

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="ec2",
        exp_prefix="multi_agent_12"
        # plot=True,
    )