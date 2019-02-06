"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import MinicityTrainingRouter_5
from flow.core.traffic_lights import TrafficLights
import numpy as np

np.random.seed(204)


def minicity_example(render=None,
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
    sumo_params = SumoParams(sim_step=0.5,
                             emission_path='./data/')

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

    vehicles = Vehicles()
    # # upper right

    # upper left
    edge_starts = ['e_12', 'e_18', 'e_19', 'e_24', 'e_45', 'e_43',
                    'e_41', 'e_88', 'e_26', 'e_34', 'e_23', 'e_5', 'e_4',
                    'e_3', 'e_25', 'e_87', 'e_40', 'e_42', 'e_44', 'e_15',
                    'e_16', 'e_20', 'e_47']

    edge_starts = list(set(edge_starts))

    # add vehicle
    vehicles.add(
        veh_id='human',
        acceleration_controller=(IDMController, {}),
        routing_controller=(MinicityTrainingRouter_5, {}),
        speed_mode='right_of_way',
        lane_change_mode='no_lat_collide',
        num_vehicles=50)

    tl_logic = TrafficLights(baseline=False)

    nodes = ["n_i1", 'n_i2', 'n_i3', "n_i4", 'n_i6', 'n_i7', 'n_i8']
    phases = [{"duration": "20", "state": "GGGGrrGGGGrr"},
              {"duration": "4", "state": "yyyGrryyGyrr"},
              {"duration": "20", "state": "GrrGGGGrrGGG"},
              {"duration": "4", "state": "GrryyyGrryyy"}]

    #top left traffic light
    phases_2 = [{"duration": "20", "state": "GGGrGG"},
              {"duration": "4", "state": "yyyryy"},
              {"duration": "10", "state": "rrGGGr"},
              {"duration": "4", "state": "rryyyr"}]

    #center traffic light
    phases_3 = [{"duration": "20", "state": "GGGGGrrrGGGGGrrr"},
                {"duration": "4", "state": "yyyyyrrryyyyyrrr"},
                {"duration": "20", "state": "GrrrGGGGGrrrGGGG"},
                {"duration": "4", "state": "yrrryyyyyrrryyyy"}]

    #bottom right traffic light
    phases_6 = [{"duration": "20", "state": "GGGGGrr"},
                {"duration": "4", "state": "yyGGGrr"},
                {"duration": "20", "state": "GrrrGGG"},
                {"duration": "4", "state": "Grrryyy"}]

    #top right traffic light
    phases_8 = [{"duration": "20", "state": "GrrrGGG"},
                {"duration": "4", "state": "Grrryyy"},
                {"duration": "20", "state": "GGGGGrr"},
                {"duration": "4", "state": "yyGGGrr"}]

    for node_id in nodes:
        if node_id == 'n_i2':
            tl_logic.add(node_id, phases=phases_2,
                         tls_type="actuated", programID=1)
        elif node_id == 'n_i3':
            tl_logic.add(node_id, phases=phases_3,
                         tls_type="actuated", programID=1)
        elif node_id == 'n_i6':
            tl_logic.add(node_id, phases=phases_6,
                         tls_type="actuated", programID=1)
        elif node_id == 'n_i8':
            tl_logic.add(node_id, phases=phases_8,
                         tls_type="actuated", programID=1)
        else:
            tl_logic.add(node_id, phases=phases,
                         tls_type="actuated", programID=1)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()

    additional_net_params['traffic_lights'] = True

    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing='random',
        edges_distribution=edge_starts,
        min_gap=2)
    # initial_config = InitialConfig(
    #     spacing="random",
    #     min_gap=5
    # )
    scenario = MiniCityScenario(
        name='minicity',
        vehicles=vehicles,
        initial_config=initial_config,
        net_params=net_params,
        traffic_lights=tl_logic)

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    # There are six modes of pyglet rendering:
    # No rendering: minicity_example(render=False)
    # SUMO-GUI rendering: minicity_example(render=True)
    # Static grayscale rendering: minicity_example(render="gray")
    # Dynamic grayscale rendering: minicity_example(render="dgray")
    # Static RGB rendering: minicity_example(render="rgb")
    # Dynamic RGB rendering: minicity_example(render="drgb")
    import time
    for _ in range(100):
        # t = time.time()
        exp = minicity_example(render=True,
                               save_render=False,
                               sight_radius=50,
                               pxpm=3,
                               show_radius=True)

        # run for a set number of rollouts / time steps
        exp.run(1, 7200, convert_to_csv=True)
        # print(time.time() - t)
