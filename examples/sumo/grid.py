"""Grid example."""
from flow.controllers.routing_controllers import GridRouter
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams
from flow.core.vehicles import Vehicles
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
    n = 2
    m = 3
    num_cars_left = 20
    num_cars_right = 20
    num_cars_top = 20
    num_cars_bot = 20
    tot_cars = (num_cars_left + num_cars_right) * m \
        + (num_cars_top + num_cars_bot) * n

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

    tl_logic = TrafficLights(baseline=False)
    phases = [{
        "duration": "31",
        "minDur": "8",
        "maxDur": "45",
        "state": "GrGr"
    }, {
        "duration": "6",
        "minDur": "3",
        "maxDur": "6",
        "state": "yryr"
    }, {
        "duration": "31",
        "minDur": "8",
        "maxDur": "45",
        "state": "rGrG"
    }, {
        "duration": "6",
        "minDur": "3",
        "maxDur": "6",
        "state": "ryry"
    }]
    tl_logic.add("center0", phases=phases, programID=1)
    tl_logic.add("center1", phases=phases, programID=1)
    tl_logic.add("center2", tls_type="actuated", phases=phases, programID=1)

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 1,
        "vertical_lanes": 1
    }
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

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
    exp.run(1, 1500)
