"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
#from flow.controllers.routing_controllers import MinicityRouter
from flow.core.traffic_lights import TrafficLights
from flow.controllers.routing_controllers import MinicityTrainingRouter_9
import numpy as np
seed=204
np.random.seed(seed)


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

    sumo_car_following_params = SumoCarFollowingParams(decel=7.5)

    vehicles = Vehicles()
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(MinicityRouter, {}),
        speed_mode=31,
        lane_change_mode="aggressive",
        initial_speed=0,
        num_vehicles=110,
        sumo_car_following_params=sumo_car_following_params)
    # vehicles.add(
    #     veh_id="rl",
    #     acceleration_controller=(RLController, {}),
    #     routing_controller=(MinicityRouter, {}),
    #     speed_mode="right_of_way",
    #     initial_speed=0,
    #     num_vehicles=10)

    tl_logic = TrafficLights(baseline=False)

    nodes = ["n_i1", 'n_i2', 'n_i3', "n_i4", 'n_i6', 'n_i7', 'n_i8', 'n_m3']
    phases = [{"duration": "15", "state": "GGGGrrGGGGrr"},
              {"duration": "5", "state": "yyyGrryyGyrr"},
              {"duration": "15", "state": "GrrGGGGrrGGG"},
              {"duration": "15", "state": "GrryyyGrryyy"}]

    # merge
    phases_m3 = [{"duration": "15", "state": "GGrG"},
                 {"duration": "5", "state": "yGry"},
                 {"duration": "15", "state": "rGGr"},
                 {"duration": "5", "state": "rGyr"}]

    # top left traffic light
    phases_2 = [{"duration": "15", "state": "GGGrGG"},
                {"duration": "5", "state": "yyyryy"},
                {"duration": "15", "state": "rrGGGr"},
                {"duration": "5", "state": "rryyyr"}]

    # center traffic light
    phases_3 = [{"duration": "15", "state": "GGGGGrrrGGGGGrrr"},
                {"duration": "5", "state": "yyyyyrrryyyyyrrr"},
                {"duration": "15", "state": "GrrrGGGGGrrrGGGG"},
                {"duration": "5", "state": "yrrryyyyyrrryyyy"}]

    # bottom right traffic light
    phases_6 = [{"duration": "15", "state": "GGGGGrr"},
                {"duration": "5", "state": "yyGGGrr"},
                {"duration": "15", "state": "GrrrGGG"},
                {"duration": "5", "state": "Grrryyy"}]

    # top right traffic light
    phases_8 = [{"duration": "15", "state": "GrrrGGG"},
                {"duration": "5", "state": "Grrryyy"},
                {"duration": "15", "state": "GGGGGrr"},
                {"duration": "5", "state": "yyGGGrr"}]

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
        elif node_id == 'n_m3':
            tl_logic.add(node_id, phases=phases_m3,
                         tls_type="actuated", programID=1)
        else:
            tl_logic.add(node_id, phases=phases,
                         tls_type="actuated", programID=1)
    # sumo_params.sim_step = 0.2

    vehicles = Vehicles()

    # section 1: bottom_left
    section_1 = {'e_2': [('section1_track', 3), ('idm', 2)],
                 'e_3': [('idm', 5)],
                 'e_25': [('idm', 4)],
                 'e_31': [('idm', 2)],
                 'e_39': [('idm', 3)],
                 'e_41': [('idm', 3)]}
    # section 2: center_left
    section_2 = {'e_3': [('section2_track', 3), ('idm', 4)],
                 'e_26': [('idm', 10)],
                 'e_66': [('idm', 3)],
                 'e_87': [('idm', 3)]}
    # section 3: center_center
    section_3 = {'e_41': [('section3_track', 5), ('idm', 1)],
                 'e_3': [('idm', 8)],
                 'e_25': [('idm', 8)],
                 'e_38': [('idm', 8)],
                 'e_54': [('idm', 6)],
                 'e_87': [('idm', 2)]}
    # section 4: bottom_center
    section_4 = {'e_39': [('section4_track', 5), ('idm', 1)],
                 'e_63': [('idm', 4)],
                 'e_31': [('idm', 3)],
                 'e_51': [('idm', 1)],
                 'e_94': [('idm', 2)],
                 'e_29_u': [('idm', 1)]}
    # section 5: top left
    section_5 = {'e_34': [('section5_track', 4), ('idm', 1)],
                 'e_23': [('section5_track', 2), ('idm', 3)],
                 'e_13': [('idm', 2)],
                 'e_14': [('idm', 3)],
                 'e_27': [('idm', 2)],
                 'e_6': [('idm', 1)],
                 'e_12': [('idm', 2)],
                 'e_35': [('idm', 1)]}
    # section 6: right center
    section_6 = {'e_60': [('section6_track', 1), ('idm', 1)],
                 'e_50': [('section6_track', 3), ('idm', 1)],
                 'e_74': [('idm', 4)],
                 'e_67': [('idm', 3)],
                 'e_71': [('idm', 2)],
                 'e_69': [('idm', 1)],
                 'e_64': [('idm', 2)]}
    # section 7: top center
    section_7 = {'e_42': [('section7_track', 3), ('idm', 1)],
                 'e_44': [('section7_track', 2), ('idm', 1)],
                 'e_79': [('idm', 8)],
                 'e_59': [('idm', 3)],
                 'e_24': [('idm', 1)],
                 'e_33': [('idm', 3)],
                 'e_47': [('idm', 2)],
                 'e_86': [('idm', 2)],
                 'e_22': [('idm', 3)],
                 'e_40': [('idm', 2)]}
    # top right
    section_8 = {'e_84': [('idm', 1)],
                 'e_73': [('section8_track', 3)],
                 'e_77': [('idm', 1)],
                 'e_56': [('idm', 3)],
                 'e_89': [('idm', 1)],
                 'e_80': [('idm', 3)],
                 'e_83': [('idm', 2)],
                 'e_82': [('idm', 1)],
                 'e_90': [('idm', 1)],
                 'e_78': [('idm', 1)],
                 'e_76': [('idm', 2)],
                 # 'e_86': [('idm', 4)],
                 'e_75': [('idm', 1)]}

    experiment = section_2
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
            acceleration_controller=(IDMController, {}),
            routing_controller=(MinicityTrainingRouter_9, {}),
            speed_mode='no_collide',
            lane_change_mode='strategic',
            num_vehicles=v_num)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()

    additional_net_params['traffic_lights'] = True
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing='random',
        edges_distribution=experiment,
        # min_gap=2
    )
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
    exp = minicity_example(render=True,
                           save_render=False,
                           sight_radius=20,
                           pxpm=3,
                           show_radius=False)

    # run for a set number of rollouts / time steps
    exp.run(1, 1000)
