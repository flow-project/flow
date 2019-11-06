"""Traffic Light Grid example."""
from flow.controllers import GridRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import InFlows
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks import TrafficLightGridNetwork


def gen_edges(col_num, row_num):
    """Generate the names of the outer edges in the traffic light grid network.

    Parameters
    ----------
    col_num : int
        number of columns in the traffic light grid
    row_num : int
        number of rows in the traffic light grid

    Returns
    -------
    list of str
        names of all the outer edges
    """
    edges = []

    x_max = col_num + 1
    y_max = row_num + 1

    def new_edge(from_node, to_node):
        return str(from_node) + "--" + str(to_node)

    # Build the horizontal edges
    for y in range(1, y_max):
        for x in [0, x_max - 1]:
            left_node = "({}.{})".format(x, y)
            right_node = "({}.{})".format(x + 1, y)
            edges += new_edge(left_node, right_node)
            edges += new_edge(right_node, left_node)

    # Build the vertical edges
    for x in range(1, x_max):
        for y in [0, y_max - 1]:
            bottom_node = "({}.{})".format(x, y)
            top_node = "({}.{})".format(x, y + 1)
            edges += new_edge(bottom_node, top_node)
            edges += new_edge(top_node, bottom_node)

    return edges


def get_flow_params(col_num, row_num, additional_net_params):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    col_num : int
        number of columns in the traffic light grid
    row_num : int
        number of rows in the traffic light grid
    additional_net_params : dict
        network-specific parameters that are unique to the traffic light grid

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    initial = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='human',
            edge=outer_edges[i],
            probability=0.25,
            departLane='free',
            departSpeed=20)

    net = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    return initial, net


def get_non_flow_params(enter_speed, add_net_params):
    """Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params: dict
        additional network-specific parameters (unique to the traffic light grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    additional_init_params = {'enter_speed': enter_speed}
    initial = InitialConfig(
        spacing='custom', additional_params=additional_init_params)
    net = NetParams(additional_params=add_net_params)

    return initial, net


def traffic_light_grid_example(render=None, use_inflows=False):
    """
    Perform a simulation of vehicles on a traffic light grid.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution
    use_inflows : bool, optional
        set to True if you would like to run the experiment with inflows of
        vehicles from the edges, and False otherwise

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles and balanced traffic lights on a traffic light grid.
    """
    v_enter = 10
    inner_length = 300
    long_length = 500
    short_length = 300
    n_rows = 2
    n_columns = 3
    num_cars_left = 20
    num_cars_right = 20
    num_cars_top = 20
    num_cars_bot = 20
    tot_cars = (num_cars_left + num_cars_right) * n_columns \
        + (num_cars_top + num_cars_bot) * n_rows

    grid_array = {
        "short_length": short_length,
        "inner_length": inner_length,
        "long_length": long_length,
        "row_num": n_rows,
        "col_num": n_columns,
        "cars_left": num_cars_left,
        "cars_right": num_cars_right,
        "cars_top": num_cars_top,
        "cars_bot": num_cars_bot
    }

    sim_params = SumoParams(sim_step=0.1, render=True)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
        ),
        num_vehicles=tot_cars)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    tl_logic = TrafficLightParams(baseline=False)
    phases = [{
        "duration": "15",
        "minDur": "8",
        "maxDur": "7",
        "state": "ggggggggggggggggggrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrggg"
    }, {
        "duration": "6",
        "minDur": "3",
        "maxDur": "6",
        "state": "ggggggggggggggggggrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"
    }, {
        "duration": "31",
        "minDur": "8",
        "maxDur": "45",
        "state": "rrrgggyyyrrrgggyyyrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"
    }, {
        "duration": "6",
        "minDur": "3",
        "maxDur": "6",
        "state": "rrrgggyyyrrrgggyyyrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"
    }]

    # Here's an example of how you can manually set traffic lights
    tl_logic.add("(1.1)", phases=phases, tls_type="actuated")
    tl_logic.add("(2.1)", phases=phases, tls_type="actuated")
    tl_logic.add("(3.1)", phases=phases, tls_type="actuated")

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 3,
        "vertical_lanes": 3
    }

    if use_inflows:
        initial_config, net_params = get_flow_params(
            col_num=n_columns,
            row_num=n_rows,
            additional_net_params=additional_net_params)
    else:
        initial_config, net_params = get_non_flow_params(
            enter_speed=v_enter,
            add_net_params=additional_net_params)

    network = TrafficLightGridNetwork(
        name="grid-intersection",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)


    env = AccelEnv(env_params, sim_params, network)

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = traffic_light_grid_example()
    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
