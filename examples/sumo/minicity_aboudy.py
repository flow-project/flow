"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.controllers import BaseRouter
from flow.core.experiment import SumoExperiment # Modified from Experiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
#from flow.core.params import VehicleParams
from flow.core.vehicles import Vehicles # Modified from VehicleParams
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
import numpy as np
import random

np.random.seed(204)

# Denotes the route choice behavior of vehicles on an edge.
#
# The key is the name of the starting edge of the vehicle, and the element if
# the name of the next edge the vehicle will take.
#
# 1. If the element is a string  name of an edge, the vehicle will always take
#    that edge as a next edge.
# 2. If the element is a list of strings, the vehicle will uniformly choose
#    among all written edges.
# 3. If the element is a list of tuples, the specific next edges are chosen
#    using the weighting specific by the second element.
NEW_EDGE = {
    'e_40': 'e_42',
    'e_42': 'e_44',
    'e_44': ['e_49', 'e_46'],
    'e_49': 'e_58',
    'e_58': 'e_76',
    'e_76': 'e_74',
    'e_74': ['e_70', 'e_72'],
    'e_70': 'e_61',
    'e_61': 'e_54',
    'e_54': 'e_40',
    'e_46': 'e_48',
    'e_48': 'e_78',
    'e_78': 'e_76',
    'e_72': 'e_68',
    'e_68': 'e_66',
    'e_66': 'e_63',
    'e_63': 'e_94',
    'e_94': 'e_52',
    'e_52': 'e_38',
    'e_38': 'e_40',
}


class MinicityRouter(BaseRouter):
    """A router used to continuously re-route vehicles in minicity scenario.

    This class allows the vehicle to pick a random route at junctions.
    """

    def __init__(self, veh_id, router_params):
        self.prev_edge = None
        super().__init__(veh_id, router_params)

    def choose_route(self, env):
        """See parent class."""
        next_edge = None
        edge = env.vehicles.get_edge(self.veh_id) # modified from env.k.vehicle
        if edge[0] == 'e_63':
            return ['e_63', 'e_94', 'e_52']
        if edge not in NEW_EDGE or edge == self.prev_edge:
            next_edge = None
        elif type(NEW_EDGE[edge]) == str:
            next_edge = NEW_EDGE[edge]
        elif type(NEW_EDGE[edge]) == list:
            if type(NEW_EDGE[edge][0]) == str:
                next_edge = random.choice(NEW_EDGE[edge])
            else:
                pass
        self.prev_edge = edge
        if next_edge is None:
            return None
        else:
            return [edge, next_edge]


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
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on the minicity scenario.
    """
    sim_params = SumoParams(sim_step=0.25)

    if render is not None:
        sim_params.render = render

    if save_render is not None:
        sim_params.save_render = save_render

    if sight_radius is not None:
        sim_params.sight_radius = sight_radius

    if pxpm is not None:
        sim_params.pxpm = pxpm

    if show_radius is not None:
        sim_params.show_radius = show_radius

    vehicles = Vehicles() # modified from VehicleParams
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        routing_controller=(MinicityRouter, {}),
        # car_following_params=SumoCarFollowingParams(
        #     speed_mode=1,
        # ),
        # lane_change_params=SumoLaneChangeParams(
        #     lane_change_mode="strategic",
        # ),
        initial_speed=0,
        num_vehicles=50)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(MinicityRouter, {}),
        # car_following_params=SumoCarFollowingParams(
        #     speed_mode="strategic",
        # ),
        initial_speed=0,
        num_vehicles=0)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing="random",
        min_gap=5,
        edges_distribution=list(NEW_EDGE.keys())
    )
    scenario = MiniCityScenario(
        name="minicity",
        vehicles=vehicles,
        initial_config=initial_config,
        net_params=net_params)

    env = AccelEnv(env_params, sim_params, scenario)

    return SumoExperiment(env, scenario) # modified from Experiement(), added scenario param


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
                           sight_radius=30,
                           pxpm=3,
                           show_radius=True)

    # run for a set number of rollouts / time steps
    exp.run(1, 750)
