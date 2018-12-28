"""Example of modified intersection network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController, IDMController, ConstAccController
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig,\
    SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.intersection import IntersectionScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import IntersectionRouter
import numpy as np
seed=204
np.random.seed(seed)


def intersection_example(render=None,
                     save_render=None,
                     sight_radius=None,
                     pxpm=None,
                     show_radius=None):
    """
    Perform a simulation of vehicles on modified minicity of University of
    Delaware.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on the minicity scenario.
    """
    sumo_params = SumoParams(render=False,seed=seed)

    if render is not None:
        sumo_params.render = render

    if save_render is not None:
        sumo_params.save_render = save_render

    if sight_radius is not None:
        sumo_params.sight_radius = sight_radius

    if pxpm is not None:
        sumo_params.pxpm = pxpm

    if show_radius is not None:
        sumo_params.show_radius = show_radius

    # sumo_params.sim_step = 0.2

    vehicles = Vehicles()

    experiment = {'e_1': [('rl', 10)],
                  'e_3': [('rl', 10)],
                  'e_5': [('rl', 10)],
                  'e_7': [('rl', 10)]}
    vehicle_data = {}
    # get all different vehicle types
    for _, pairs in experiment.items():
        for pair in pairs:
            cur_num = vehicle_data.get(pair[0], 0)
            vehicle_data[pair[0]] = cur_num + pair[1]

    # add vehicle
    for v_type, v_num in vehicle_data.items():
        vehicles.add(
            veh_id=v_type,
            acceleration_controller=(ConstAccController, {}),
            routing_controller=(IntersectionRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                min_gap=0,
            ),
            speed_mode=0,#'no_collide',
            lane_change_mode=0,
            num_vehicles=v_num)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing='uniform',
        edges_distribution=experiment,
    )
    scenario = IntersectionScenario(
        name='intersection',
        vehicles=vehicles,
        initial_config=initial_config,
        net_params=net_params)

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    exp = intersection_example(render='drgb',
                           save_render=False,
                           sight_radius=20,
                           pxpm=3,
                           show_radius=False)

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
