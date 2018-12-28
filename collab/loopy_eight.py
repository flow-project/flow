"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.loopy_eight import LoopyEightScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import LoopyEightRouter
import numpy as np
seed=204
np.random.seed(seed)

def loopy_eight_example(render=None,
                     save_render=None,
                     sight_radius=None,
                     pxpm=None,
                     show_radius=None):

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

    vehicles.add(
        veh_id = 'rl',
        acceleration_controller = (RLController, {}),
        routing_controller = (LoopyEightRouter, {}),
        speed_mode = 23,#'no_collide',
        lane_change_mode = 'strategic',
        num_vehicles = 50)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing='random',
        edges_distribution='all',
        min_gap=5
    )

    scenario = LoopyEightScenario(
        name='loopy_eight',
        vehicles=vehicles,
        initial_config=initial_config,
        net_params=net_params)

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    exp = loopy_eight_example(render='drgb',
                           save_render=False,
                           sight_radius=20,
                           pxpm=3,
                           show_radius=False)

    # run for a set number of rollouts / time steps
    exp.run(1, 3000)
