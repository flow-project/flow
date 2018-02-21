"""
(description)
"""
from flow.controllers.car_following_models import IDMController
from flow.scenarios.grid.gen import SimpleGridGenerator
from flow.scenarios.grid.grid_scenario import SimpleGridScenario
from flow.core.experiment import SumoExperiment
from flow.envs.loop_accel import AccelEnv
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.controllers.routing_controllers import GridRouter


def grid_example(sumo_binary=None):
    inner_length = 300
    long_length = 500
    short_length = 300
    n = 2
    m = 2
    num_cars_left = 20
    num_cars_right = 20
    num_cars_top = 20
    num_cars_bot = 20
    # tot_cars = num_cars_edge * 2 * n + num_cars_edge * 2 * m

    grid_array = {"short_length": short_length, "inner_length": inner_length,
                  "long_length": long_length, "row_num": n, "col_num": m,
                  "cars_left": num_cars_left, "cars_right": num_cars_right,
                  "cars_top": num_cars_top, "cars_bot": num_cars_bot}

    sumo_params = SumoParams(sim_step=0.1,
                             sumo_binary="sumo-gui")

    if sumo_binary is not None:
        sumo_params.sumo_binary = sumo_binary

    vehicles = Vehicles()
    vehicles.add(veh_id="human",
                 acceleration_controller=(IDMController, {}),
                 routing_controller=(GridRouter, {}),
                 num_vehicles=1)

    additional_env_params = {"target_velocity": 8}
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = {"length": 200, "lanes": 2, "speed_limit": 35,
                             "resolution": 40, "grid_array": grid_array,
                             "horizontal_lanes": 1, "vertical_lanes": 1,
                             "traffic_lights": True}
    net_params = NetParams(no_internal_links=False,
                           additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="custom")

    scenario = SimpleGridScenario(name="grid-intersection",
                                  generator_class=SimpleGridGenerator,
                                  vehicles=vehicles,
                                  net_params=net_params,
                                  initial_config=initial_config)

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = grid_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 15000)
