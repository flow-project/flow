"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import MinicityTrainingRouter_4
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
    # upper right
    edge_starts = ['e_80', 'e_83', 'e_82', 'e_79', 'e_47', 'e_49', 'e_55',
                   'e_56', 'e_89', 'e_45', 'e_43', 'e_41', 'e_50', 'e_60',
                   'e_69', 'e_73', 'e_75', 'e_86', 'e_59', 'e_48', 'e_81',
                   'e_84', 'e_85', 'e_90', 'e_62', 'e_57', 'e_46', 'e_76',
                   'e_76', 'e_74', 'e_70', 'e_61', 'e_54', 'e_40', 'e_42',
                   'e_44']
    # bottom-left
    edge_starts += ['e_25', 'e_30', 'e_31', 'e_32', 'e_21', 'e_8_u', 'e_9',
                    'e_10', 'e_11', 'e_87', 'e_39', 'e_37', 'e_29_u', 'e_92',
                    'e_7', 'e_8_b', 'e_10']
    # upper left
    edge_starts += ['e_12', 'e_18', 'e_19', 'e_24', 'e_45', 'e_43',
                    'e_41', 'e_88', 'e_26', 'e_34', 'e_23', 'e_5', 'e_4',
                    'e_3', 'e_25', 'e_87', 'e_40', 'e_42', 'e_44', 'e_15',
                    'e_16', 'e_20', 'e_47', 'e_46']
    # bottom right corner
    edge_starts += ['e_50', 'e_60', 'e_69', 'e_72', 'e_68', 'e_66', 'e_63',
                    'e_94', 'e_52', 'e_38']
    # bottom half outer loop
    edge_starts += ['e_67', 'e_71', 'e_70', 'e_61', 'e_54', 'e_88', 'e_26',
                    'e_2', 'e_1', 'e_7', 'e_17', 'e_28_b', 'e_36', 'e_93',
                    'e_53', 'e_64']
    # bottom right inner loop
    edge_starts += ['e_50', 'e_60', 'e_69', 'e_72', 'e_68', 'e_66', 'e_63',
                    'e_94', 'e_52', 'e_38']

    edge_starts = list(set(edge_starts))

    # add vehicle
    vehicles.add(
        veh_id='human',
        acceleration_controller=(IDMController, {}),
        routing_controller=(MinicityTrainingRouter_4, {}),
        speed_mode='no_collide',
        lane_change_mode='strategic',
        num_vehicles=100)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
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
    import time
    for _ in range(100):
        # t = time.time()
        exp = minicity_example(render='drgb',
                               save_render=False,
                               sight_radius=50,
                               pxpm=3,
                               show_radius=True)

        # run for a set number of rollouts / time steps
        exp.run(1, 7200, convert_to_csv=True)
        # print(time.time() - t)
