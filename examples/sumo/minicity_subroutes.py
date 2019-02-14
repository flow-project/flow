"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import *
from flow.core.traffic_lights import TrafficLights
import numpy as np
from enum import Enum

np.random.seed(204)


# Definitions of subnetworks
class SubRoute(Enum):
    ALL =  0
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM = 3

subroute_edge_starts = [
                    # Full network
                    None,

                    # Top left
                    ['e_12', 'e_18', 'e_19', 'e_24', 'e_45', 'e_43',
                    'e_41', 'e_88', 'e_26', 'e_34', 'e_23', 'e_5', 'e_4',
                    'e_3', 'e_25', 'e_87', 'e_40', 'e_42', 'e_44', 'e_15',
                    'e_16', 'e_20', 'e_47'],

                    # Top right
                    ['e_80', 'e_83', 'e_82', 'e_79', 'e_47', 'e_49', 'e_55',
                    'e_56', 'e_89', 'e_45', 'e_43', 'e_41', 'e_50', 'e_60',
                    'e_69', 'e_73', 'e_75', 'e_59', 'e_48', 'e_81',
                    'e_84', 'e_85', 'e_90', 'e_62', 'e_57', 'e_46', 'e_76',
                    'e_76', 'e_74', 'e_70', 'e_61', 'e_54', 'e_40', 'e_42',
                    'e_44'],

                    # Bottom (right corner, half outer loop, right inner loop)
                    ['e_50', 'e_60', 'e_69', 'e_72', 'e_68', 'e_66', 'e_63',
                    'e_94', 'e_52', 'e_38', 
                    'e_67', 'e_71', 'e_70', 'e_61', 'e_54', 'e_88', 'e_26',
                    'e_2', 'e_1', 'e_7', 'e_17', 'e_28_b', 'e_36', 'e_93',
                    'e_53', 'e_64',
                    'e_50', 'e_60', 'e_69', 'e_72', 'e_68', 'e_66', 'e_63',
                    'e_94', 'e_52', 'e_38'],

                    ]

subroute_controllers = [
                    MinicityRouter, # full network
                    MinicityTrainingRouter_5, # top left
                    MinicityTrainingRouter_4, # top right
                    MinicityTrainingRouter_6, # bottom
                    ]




def minicity_example(render=None,
                     save_render=None,
                     sight_radius=None,
                     pxpm=None,
                     show_radius=None,
                     subroute=SubRoute.ALL):
    """
    Perform a simulation of vehicles on modified minicity of University of
    Delaware.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use sumo's gui during execution
    subroute: optional, specifies whether to constrict car routes to 
        pre-defined subnetworks. 
        Pass in as SubRoute.TOP_LEFT, SubRoute.TOP_RIGHT, or SubRoute.BOTTOM

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

    # add vehicle
    vehicles.add(
        veh_id='human',
        acceleration_controller=(IDMController, {}),
        routing_controller=(subroute_controllers[subroute.value], {}),
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
    
    if subroute == SubRoute.ALL:
        initial_config = InitialConfig(
            spacing="random",
            min_gap=5
        )
    else:
        # Define subroute edges
        edge_starts = list(set(subroute_edge_starts[subroute.value]))

        initial_config = InitialConfig(
            spacing='random',
            edges_distribution=edge_starts,
            min_gap=2
        )


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
                               show_radius=True,
                               subroute=SubRoute.ALL) # Change this line to specify subnetwork
                               # Pass in as SubRoute.ALL, SubRoute.TOP_LEFT, SubRoute.TOP_RIGHT, or SubRoute.BOTTOM

        # run for a set number of rollouts / time steps
        exp.run(1, 7200, convert_to_csv=True)
        # print(time.time() - t)
