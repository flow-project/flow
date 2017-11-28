''' Used to test out a mixed environment with an IDM controller and
another type of car, in this case our drunk driver class. One lane. 

Variables:
    sumo_params {dict} -- [Pass time step, safe mode is on or off]
    sumo_binary {str} -- [Use either sumo-gui or sumo for visual or non-visual]
    type_params {dict} -- [Types of cars in the system. 
    Format {"name": (number, (Model, {params}), (Lane Change Model, {params}), initial_speed)}]
    env_params {dict} -- [Params for reward function]
    net_params {dict} -- [Params for network.
                            length: road length
                            lanes
                            speed limit
                            resolution: number of edges comprising ring
                            net_path: where to store net]
    cfg_params {dict} -- [description]
    initial_config {dict} -- [shuffle: randomly reorder cars to start experiment
                                spacing: if gaussian, add noise in start positions
                                bunching: how close to place cars at experiment start]
    scenario {[type]} -- [Which road network to use]
'''
import flow.core.config as flow_config
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.controllers.rlcarfollowingcontroller import RLCarFollowingController
from flow.controllers.routing_controllers import *
from flow.core.params import *
from flow.core.vehicles import Vehicles
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


def run_task(*_):

    sumo_params = SumoParams(time_step=0.1, sumo_binary="sumo-gui")

    vehicles = Vehicles()

    human_cfm_params = SumoCarFollowingParams(carFollowModel="IDM", sigma=1.0, tau=3.0, speedDev=0.1, minGap=3.0)
    human_lc_params = SumoLaneChangeParams(lcKeepRight=0, lcAssertive=0.5,
                                           lcSpeedGain=1.5, lcSpeedGainRight=1.0, model="SL2015")
    vehicles.add_vehicles("human", (SumoCarFollowingController, {}), (SumoLaneChangeController, {}),
                          (ContinuousRouter, {}),
                          0, 14,
                          lane_change_mode="execute_all",
                          sumo_car_following_params=human_cfm_params,
                          sumo_lc_params=human_lc_params,
                          )

    aggressive_cfm_params = SumoCarFollowingParams(carFollowModel="IDM", speedFactor=1.75, tau=0.1, minGap=0.5)
    vehicles.add_vehicles("aggressive-human", (SumoCarFollowingController, {}),
                          (SafeAggressiveLaneChanger, {"target_velocity": 22.25, "threshold": 0.7}),
                          (ContinuousRouter, {}), 0, 1,
                          lane_change_mode="custom", custom_lane_change_mode=0b0100000000,
                          sumo_car_following_params=aggressive_cfm_params)

    vehicles.add_vehicles("rl", (RLCarFollowingController, {}), None, (ContinuousRouter, {}), 0, 3,
                          lane_change_mode="custom", custom_lane_change_mode=512)

    env_params = EnvParams(additional_params={"target_velocity": 15, "num_steps": 1000}, lane_change_duration=0)

    additional_net_params = {"length": 300, "lanes": 3, "speed_limit": 15, "resolution": 40}
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="custom", lanes_distribution=3, shuffle=False)

    scenario = LoopScenario("3-lane-aggressive-driver", CircleGenerator, vehicles, net_params,
                            initial_config)

    env_name = "ShepherdingEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params, initial_config, scenario)
    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
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
        batch_size=15000,
        max_path_length=horizon,
        n_itr=2000,
    )
    algo.train()

run_experiment_lite(
    run_task,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used,
    exp_prefix="shepherding",
    # Number of parallel workers for sampling
    n_parallel=16,
    python_command=flow_config.PYTHON_COMMAND,
    mode="ec2",
    seed=60,
    # n_parallel=1,
    # python_command="/Users/kanaad/anaconda3/envs/flow/bin/python",
    # mode="local"
)
