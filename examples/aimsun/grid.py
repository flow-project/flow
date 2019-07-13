"""Grid example."""
from flow.core.experiment import Experiment
from flow.core.params import AimsunParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.grid import SimpleGridScenario


def grid_example(render=None):
    """
    Perform a simulation of vehicles on a grid.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles and balanced traffic lights on a grid.
    """
    inner_length = 300
    long_length = 500
    short_length = 300
    N_ROWS = 2
    N_COLUMNS = 3
    num_cars_left = 20
    num_cars_right = 20
    num_cars_top = 20
    num_cars_bot = 20
    tot_cars = (num_cars_left + num_cars_right) * N_COLUMNS \
        + (num_cars_top + num_cars_bot) * N_ROWS

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

    sim_params = AimsunParams(sim_step=0.5, render=True)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        num_vehicles=tot_cars)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    tl_logic = TrafficLightParams(baseline=False)
    phases = [{
        "duration": "31",
        "minDur": "8",
        "maxDur": "45",
        "yellow": "3",
        "state": "GGGrrrGGGrrr"
    }, {
        "duration": "6",
        "minDur": "3",
        "maxDur": "6",
        "yellow": "3",
        "state": "yyyrrryyyrrr"
    }, {
        "duration": "31",
        "minDur": "8",
        "maxDur": "45",
        "yellow": "3",
        "state": "rrrGGGrrrGGG"
    }, {
        "duration": "6",
        "minDur": "3",
        "maxDur": "6",
        "yellow": "3",
        "state": "rrryyyrrryyy"
    }]
    tl_logic.add("center0", phases=phases, programID=1)
    tl_logic.add("center1", phases=phases, programID=1)
    tl_logic.add("center2", tls_type="actuated", phases=phases, programID=1)
    tl_logic.add("center3", phases=phases, programID=1)
    tl_logic.add("center4", phases=phases, programID=1)
    tl_logic.add("center5", tls_type="actuated", phases=phases, programID=1)

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 1,
        "vertical_lanes": 1
    }
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    initial_config = InitialConfig(spacing='custom')

    scenario = SimpleGridScenario(
        name="grid-intersection",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)

    env = AccelEnv(env_params, sim_params, scenario, simulator='aimsun')

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = grid_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
