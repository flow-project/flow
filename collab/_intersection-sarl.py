"""Implementation of MIT slot-based framework BATCH for two-lanes scenario
as the intersection's state-of-art baseline."""
from flow.controllers import RLController, IDMController, ConstAccController,\
    SumoCarFollowingController, SumoLaneChangeController,\
    RandomConstAccController, RandomLaneChanger, StaticLaneChanger
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig,\
    SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.envs.intersection_baseline import IntersectionBaseline, ADDITIONAL_ENV_PARAMS
from flow.scenarios.intersection import IntersectionScenario, ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import IntersectionRandomRouter
from flow.core.params import InFlows
import numpy as np
seed=204
np.random.seed(seed)

def intersection_example(render=None,
                     save_render=None,
                     sight_radius=None,
                     pxpm=None,
                     show_radius=None):
    """
    Perform a simulation of vehicles on the two-lane intersection scenario.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of BATCH as the
        baseline performance on the intersection scenario.
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

    # experiment = {'e_1_sbc+': [('autonomous', 6)],
    #               'e_3_sbc+': [('autonomous', 8)],
    #               'e_5_sbc+': [('autonomous', 7)],
    #               'e_7_sbc+': [('autonomous', 5)]}
    # vehicle_data = {}
    # # get all different vehicle types
    # for _, pairs in experiment.items():
    #     for pair in pairs:
    #         cur_num = vehicle_data.get(pair[0], 0)
    #         vehicle_data[pair[0]] = cur_num + pair[1]
    #
    # # add vehicle
    # for veh_id, veh_num in vehicle_data.items():
    #     vehicles.add(
    #         veh_id=veh_id,
    #         speed_mode=31,
    #         lane_change_mode='strategic',
    #         acceleration_controller=(ConstAccController, {}),
    #         # lane_change_controller=(StaticLaneChanger, {}),
    #         routing_controller=(IntersectionRandomRouter, {}),
    #         num_vehicles=veh_num)

    vehicles.add(veh_id='autonomous',
                 acceleration_controller=(ConstAccController, {}),
                 speed_mode=0,
                 lane_change_mode='strategic',
                 routing_controller=(IntersectionRandomRouter, {}),
                 num_vehicles=1)

    # add inflow
    inflow = InFlows()
    inflow.add(veh_type='autonomous',
               edge='e_1_inflow',
               vehs_per_hour=1000,
               departSpeed=10,
               departLane='random')
    inflow.add(veh_type='autonomous',
               edge='e_3_inflow',
               vehs_per_hour=1000,
               departSpeed=10,
               departLane='random')
    inflow.add(veh_type='autonomous',
               edge='e_5_inflow',
               vehs_per_hour=1000,
               departSpeed=10,
               departLane='random')
    inflow.add(veh_type='autonomous',
               edge='e_7_inflow',
               vehs_per_hour=1000,
               departSpeed=10,
               departLane='random')

    env_params = EnvParams(
        additional_params=ADDITIONAL_ENV_PARAMS,
    )

    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    )

    initial_config = InitialConfig(
        spacing='uniform',
        edges_distribution=['e_1_sbc+'],
    )
    # initial_config = InitialConfig()

    scenario = IntersectionScenario(
        name='intersection',
        vehicles=vehicles,
        initial_config=initial_config,
        net_params=net_params,
    )

    env = IntersectionBaseline(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    exp = intersection_example(render=True,
                               save_render=False,
                               sight_radius=20,
                               pxpm=4,
                               show_radius=False)

    # run for a set number of rollouts / time steps
    exp.run(1, 1000)
