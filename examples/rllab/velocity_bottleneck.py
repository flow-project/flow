"""Bottleneck decongestion example.

Bottleneck in which the actions are specifying a desired velocity
in a segment of space
"""
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights

from flow.scenarios.bottleneck.gen import BottleneckGenerator
from flow.scenarios.bottleneck.scenario import BottleneckScenario
from flow.controllers.lane_change_controllers import SumoLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.rlcontroller import RLController

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.ppo import PPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

SCALING = 1
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = .1
N_CPUS = 32
i = 0

sumo_params = SumoParams(
    sim_step=0.5, render=False, restart_instance=True)

vehicles = Vehicles()

vehicles.add(
    veh_id="human",
    lane_change_controller=(SumoLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    sumo_car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    sumo_lc_params=SumoLaneChangeParams(
        lane_change_mode=0,  # 1621,#0b100000101,

    ),
    num_vehicles=1 * SCALING)
vehicles.add(
    veh_id="followerstopper",
    acceleration_controller=(RLController, {
        "fail_safe": "instantaneous"
    }),
    lane_change_controller=(SumoLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    sumo_car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    sumo_lc_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1 * SCALING)

horizon = 1000
# edge name, how many segments to observe/control, whether the segment is
# controlled
controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True),
                       ("4", 2, True), ("5", 1, False)]
num_observed_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 3), ("5", 1)]
additional_env_params = {
    "target_velocity": 40,
    "disable_tb": True,
    "disable_ramp_metering": True,
    "controlled_segments": controlled_segments,
    "symmetric": False,
    "observed_segments": num_observed_segments,
    "reset_inflow": False,
    "lane_change_duration": 5,
    "max_accel": 3,
    "max_decel": 3,
    "inflow_range": [1000, 2000]
}
env_params = EnvParams(
    additional_params=additional_env_params,
    warmup_steps=40,
    sims_per_step=2,
    horizon=horizon)

flow_rate = 1500 * SCALING
print('flow rate is ', flow_rate)
env_name = "DesiredVelocityEnv"

inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="1",
    vehs_per_hour=flow_rate * (1 - AV_FRAC),
    departLane="random",
    departSpeed=10)
inflow.add(
    veh_type="followerstopper",
    edge="1",
    vehs_per_hour=flow_rate * (AV_FRAC),
    departLane="random",
    departSpeed=10)

traffic_lights = TrafficLights()
if not DISABLE_TB:
    traffic_lights.add(node_id="2")
if not DISABLE_RAMP_METER:
    traffic_lights.add(node_id="3")

additional_net_params = {"scaling": SCALING}
net_params = NetParams(
    inflows=inflow,
    no_internal_links=False,
    additional_params=additional_net_params)

initial_config = InitialConfig(
    spacing="uniform",
    min_gap=5,
    lanes_distribution=float("inf"),
    edges_distribution=["2", "3", "4", "5"])
scenario = BottleneckScenario(
    name="bay_bridge_toll",
    generator_class=BottleneckGenerator,
    vehicles=vehicles,
    net_params=net_params,
    initial_config=initial_config,
    traffic_lights=traffic_lights)


def run_task(*_):
    """Implement the run_task method needed to run experiments with rllab."""
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianGRUPolicy(env_spec=env.spec, hidden_sizes=(64, ))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=horizon * 32 * 2,
        max_path_length=horizon,
        # whole_paths=True,
        n_itr=400,
        discount=0.999,
        # step_size=0.01,
    )
    algo.train()


# for _ in range(2):
exp_tag = "VSLLaneInflowDensityLearning"  # experiment prefix
for seed in [2]:  # , 1, 5, 10, 73]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        # plot=True,
        sync_s3_pkl=True)
