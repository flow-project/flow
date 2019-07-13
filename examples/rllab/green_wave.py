"""Green wave training experiment."""
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows
from flow.core.params import SumoCarFollowingParams

from flow.controllers import SimCarFollowingController, GridRouter

from flow.scenarios.grid import SimpleGridScenario

# set to true if you would like to run the experiment with inflows of vehicles
# from the edges, and false otherwise
USE_INFLOWS = False
# inflow rate of vehicles at every edge (only if USE_INFLOWS is set to True)
EDGE_INFLOW = 300


def gen_edges(col_num, row_num):
    """Define the names of all edges in the network.

    Parameters
    ----------
    col_num : int
        number of columns of edges in the grid
    row_num : int
        number of rows of edges in the grid

    Returns
    -------
    list of str
        names of every edge to be generated.
    """
    edges = []
    for i in range(col_num):
        edges += ["left" + str(row_num) + '_' + str(i)]
        edges += ["right" + '0' + '_' + str(i)]

    # build the left and then the right edges
    for i in range(row_num):
        edges += ["bot" + str(i) + '_' + '0']
        edges += ["top" + str(i) + '_' + str(col_num)]

    return edges


def get_flow_params(v_enter, vehs_per_hour, col_num, row_num, add_net_params):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    v_enter : float
        entering speed of inflow vehicles
    vehs_per_hour : float
        vehicle inflow rate (in veh/hr)
    col_num : int
        number of columns of edges in the grid
    row_num : int
        number of rows of edges in the grid
    add_net_params : dict
        additional network-specific parameters (unique to the grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the scenario
    """
    initial_config = InitialConfig(
        spacing="custom", lanes_distribution=float("inf"), shuffle=True)

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
        additional_params=add_net_params)

    return initial_config, net_params


def get_non_flow_params(enter_speed, add_net_params):
    """Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params : dict
        additional network-specific parameters (unique to the grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the scenario
    """
    additional_init_params = {'enter_speed': enter_speed}
    initial_config = InitialConfig(
        spacing='custom', additional_params=additional_init_params)
    net_params = NetParams(
        no_internal_links=False, additional_params=add_net_params)

    return initial_config, net_params


def run_task(*_):
    """Implement the run_task method needed to run experiments with rllab."""
    V_ENTER = 30
    INNER_LENGTH = 300
    LONG_LENGTH = 100
    SHORT_LENGTH = 300
    N_ROWS = 3
    N_COLUMNS = 3
    NUM_CARS_LEFT = 1
    NUM_CARS_RIGHT = 1
    NUM_CARS_TOP = 1
    NUM_CARS_BOT = 1
    tot_cars = (NUM_CARS_LEFT + NUM_CARS_RIGHT) * N_COLUMNS \
        + (NUM_CARS_BOT + NUM_CARS_TOP) * N_ROWS

    grid_array = {
        "short_length": SHORT_LENGTH,
        "inner_length": INNER_LENGTH,
        "long_length": LONG_LENGTH,
        "row_num": N_ROWS,
        "col_num": N_COLUMNS,
        "cars_left": NUM_CARS_LEFT,
        "cars_right": NUM_CARS_RIGHT,
        "cars_top": NUM_CARS_TOP,
        "cars_bot": NUM_CARS_BOT
    }

    sim_params = SumoParams(sim_step=1, render=True)

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(SimCarFollowingController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            tau=1.1,
            max_speed=V_ENTER,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="all_checks"
        ),
        routing_controller=(GridRouter, {}),
        num_vehicles=tot_cars)

    tl_logic = TrafficLightParams(baseline=False)

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

    if USE_INFLOWS:
        initial_config, net_params = get_flow_params(
            v_enter=V_ENTER,
            vehs_per_hour=EDGE_INFLOW,
            col_num=N_COLUMNS,
            row_num=N_ROWS,
            add_net_params=additional_net_params)
    else:
        initial_config, net_params = get_non_flow_params(
            V_ENTER, additional_net_params)

    scenario = SimpleGridScenario(
        name="grid-intersection",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)

    env_name = "PO_TrafficLightGridEnv"
    pass_params = (env_name, sim_params, vehicles, env_params, net_params,
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
