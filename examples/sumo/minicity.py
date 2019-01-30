"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
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

    # sumo_params.sim_step = 0.2

    vehicles = Vehicles()

    # section 1: bottom_left
    section_1 = {'e_2': [('section1_track', 3), ('idm', 2)],
                 'e_3': [('idm', 5)],
                 'e_25': [('idm', 4)],
                 'e_31': [('idm', 2)],
                 'e_39': [('idm', 3)],
                 'e_41': [('idm', 3)]}
    # # section 2: center_left
    # section_2 = {'e_3': [('section2_track', 3), ('idm', 4)],
    #              'e_26': [('idm', 10)],
    #              'e_66': [('idm', 3)],
    #              'e_87': [('idm', 3)]}
    # # section 3: center_center
    # section_3 = {'e_41': [('section3_track', 5), ('idm', 1)],
    #              'e_3': [('idm', 8)],
    #              'e_25': [('idm', 8)],
    #              'e_38': [('idm', 8)],
    #              'e_54': [('idm', 6)],
    #              'e_87': [('idm', 2)]}
    # # section 4: bottom_center
    # section_4 = {'e_39': [('section4_track', 5), ('idm', 1)],
    #              'e_63': [('idm', 4)],
    #              'e_31': [('idm', 3)],
    #              'e_51': [('idm', 1)],
    #              'e_94': [('idm', 2)],
    #              'e_29_u': [('idm', 1)]}
    # # section 5: top left
    # section_5 = {'e_34': [('section5_track', 4), ('idm', 1)],
    #              'e_23': [('section5_track', 2), ('idm', 3)],
    #              'e_13': [('idm', 2)],
    #              'e_14': [('idm', 3)],
    #              'e_27': [('idm', 2)],
    #              'e_6': [('idm', 1)],
    #              'e_12': [('idm', 2)],
    #              'e_35': [('idm', 1)]}
    # # section 6: right center
    # section_6 = {'e_60': [('section6_track', 1), ('idm', 1)],
    #              'e_50': [('section6_track', 3), ('idm', 1)],
    #              'e_74': [('idm', 4)],
    #              'e_67': [('idm', 3)],
    #              'e_71': [('idm', 2)],
    #              'e_69': [('idm', 1)],
    #              'e_64': [('idm', 2)]}
    # # section 7: top center
    # section_7 = {'e_42': [('section7_track', 3), ('idm', 1)],
    #              'e_44': [('section7_track', 2), ('idm', 1)],
    #              'e_79': [('idm', 8)],
    #              'e_59': [('idm', 3)],
    #              'e_24': [('idm', 1)],
    #              'e_33': [('idm', 3)],
    #              'e_47': [('idm', 2)],
    #              'e_86': [('idm', 2)],
    #              'e_22': [('idm', 3)],
    #              'e_40': [('idm', 2)]}
    # # top right
    # section_8 = {'e_84': [('idm', 1)],
    #              'e_73': [('section8_track', 3)],
    #              'e_77': [('idm', 1)],
    #              'e_56': [('idm', 3)],
    #              'e_89': [('idm', 1)],
    #              'e_80': [('idm', 3)],
    #              'e_83': [('idm', 2)],
    #              'e_82': [('idm', 1)],
    #              'e_90': [('idm', 1)],
    #              'e_78': [('idm', 1)],
    #              'e_76': [('idm', 2)],
    #              # 'e_86': [('idm', 4)],
    #              'e_75': [('idm', 1)]}

    experiment = section_1
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
        net_params=net_params)

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
    exp = minicity_example(render='drgb',
                           save_render=False,
                           sight_radius=20,
                           pxpm=3,
                           show_radius=True)

    # run for a set number of rollouts / time steps
    exp.run(1, 300)
