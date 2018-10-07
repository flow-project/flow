from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows
from flow.core.params import SumoCarFollowingParams

from flow.controllers import SumoCarFollowingController, GridRouter

from flow.scenarios.grid.gen import SimpleGridGenerator
from flow.scenarios.grid.grid_scenario import SimpleGridScenario


def gen_edges(row_num, col_num):
    edges = []
    for i in range(col_num):
        edges += ["left" + str(row_num) + '_' + str(i)]
        edges += ["right" + '0' + '_' + str(i)]

    # build the left and then the right edges
    for i in range(row_num):
        edges += ["bot" + str(i) + '_' + '0']
        edges += ["top" + str(i) + '_' + str(col_num)]

    return edges


def get_flow_params(v_enter, vehs_per_hour, col_num, row_num,
                    additional_net_params):
    initial_config = InitialConfig(
        spacing="uniform", lanes_distribution=float("inf"), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type="idm",
            edge=outer_edges[i],
            vehs_per_hour=vehs_per_hour,
            departLane="free",
            departSpeed=v_enter)

    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    return initial_config, net_params


def get_non_flow_params(enter_speed, additional_net_params):
    additional_init_params = {"enter_speed": enter_speed}
    initial_config = InitialConfig(additional_params=additional_init_params)
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    return initial_config, net_params


def run_task(*_):
    """Implement the run_task method needed to run experiments with rllab."""
    v_enter = 10
    inner_length = 300
    long_length = 100
    short_length = 300
    n = 3
    m = 3
    num_cars_left = 1
    num_cars_right = 1
    num_cars_top = 1
    num_cars_bot = 1
    tot_cars = (num_cars_left + num_cars_right) * m \
        + (num_cars_bot + num_cars_top) * n

    grid_array = {
        "short_length": short_length,
        "inner_length": inner_length,
        "long_length": long_length,
        "row_num": n,
        "col_num": m,
        "cars_left": num_cars_left,
        "cars_right": num_cars_right,
        "cars_top": num_cars_top,
        "cars_bot": num_cars_bot
    }

    sumo_params = SumoParams(sim_step=1, render=True)

    vehicles = Vehicles()
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(SumoCarFollowingController, {}),
        sumo_car_following_params=SumoCarFollowingParams(
            min_gap=2.5, tau=1.1, max_speed=v_enter, speed_mode="all_checks"),
        routing_controller=(GridRouter, {}),
        num_vehicles=tot_cars)

    tl_logic = TrafficLights(baseline=False)

    additional_env_params = {
        "target_velocity": 50,
        "switch_time": 3.0,
        "num_observed": 2,
        "discrete": False,
        "tl_type": "controlled"
    }
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = {
        "speed_limit": 35,
        "grid_array": grid_array,
        "horizontal_lanes": 1,
        "vertical_lanes": 1
    }

    initial_config, net_params = get_flow_params(10, 300, n, m,
                                                 additional_net_params)

    scenario = SimpleGridScenario(
        name="grid-intersection",
        generator_class=SimpleGridGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)

    env_name = "PO_TrafficLightGridEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=40000,
        max_path_length=horizon,
        # whole_paths=True,
        n_itr=800,
        discount=0.999,
        # step_size=0.01,
    )
    algo.train()


for seed in [6]:  # , 7, 8]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode="local",  # "local_docker", "ec2"
        exp_prefix="green-wave",
        # plot=True,
    )
