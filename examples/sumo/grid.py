"""Grid example."""
from flow.controllers.routing_controllers import GridRouter
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.core.params import InFlows
from flow.core.traffic_lights import TrafficLights
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.grid import SimpleGridScenario


def grid_example(render=None):
    """
    Perform a simulation of vehicles on a grid.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles and balanced traffic lights on a grid.
    """
    inner_length = 300
    long_length = 500
    short_length = 300
    N_ROWS = 2
    N_COLUMNS = 3
    num_cars_left = 1
    num_cars_right = 1
    num_cars_top = 1
    num_cars_bot = 1
    EDGE_INFLOW = 400
    tot_cars = (num_cars_left + num_cars_right) * N_ROWS \
        + (num_cars_top + num_cars_bot) * N_COLUMNS

    # inflows of vehicles are place on all outer edges (listed here)
    outer_edges = []
    outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
    outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
    outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
    outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

    # equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
    inflow = InFlows()
    for edge in outer_edges:
        inflow.add(
            veh_type="human",
            edge=edge,
            vehs_per_hour=EDGE_INFLOW,
            departLane="free",
            departSpeed="max")

    grid_array = {
        "short_length": short_length,
        "inner_length": inner_length,
        "long_length": long_length,
        "row_num": N_ROWS,
        "col_num": N_COLUMNS,
        "cars_left": num_cars_left,
        "cars_right": num_cars_right,
        "cars_top": num_cars_top,
        "cars_bot": num_cars_bot
    }

    sumo_params = SumoParams(sim_step=0.1, render=True)

    if render is not None:
        sumo_params.render = render

    vehicles = Vehicles()
    vehicles.add(
        veh_id="human",
        routing_controller=(GridRouter, {}),
        sumo_car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
        ),
        num_vehicles=tot_cars)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    tl_logic = TrafficLights(baseline=True)

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 1,
        "vertical_lanes": 1
    }
    net_params = NetParams(
        no_internal_links=False,
        inflows=inflow,
        additional_params=additional_net_params)

    initial_config = InitialConfig()

    scenario = SimpleGridScenario(
        name="grid-intersection",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = grid_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 3000)
