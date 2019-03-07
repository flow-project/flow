"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.controllers import BaseRouter
from flow.core.experiment import SumoExperiment # Modified from Experiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, InFlows
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
#from flow.core.params import VehicleParams
from flow.core.vehicles import Vehicles # Modified from VehicleParams
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
from flow.core.traffic_lights import TrafficLights
import numpy as np
import random

from flow.scenarios.subnetworks import *
from flow.envs.minicity_env import AccelCNNSubnetEnv

from matplotlib import pyplot as plt

np.random.seed(204)


#################################################################
# MODIFIABLE PARAMETERS
#################################################################


SUBNETWORK = SubRoute.SUB4  # CHANGE THIS PARAMETER TO SELECT CURRENT SUBNETWORK

                            # Set it to SubRoute.ALL, SubRoute.TOP_LEFT, etc.

TRAFFIC_LIGHTS = True       # CHANGE THIS to True to add traffic lights to Minicity

RENDERER = True  #'drgb'        # PARAMETER. 
                            # Set to True to use default Sumo renderer, 
                            # Set to 'drgb' for Fangyu's renderer

USE_CNN = False            # Set to True to use Pixel-learning CNN agent
                            # Set to False for default vehicle speeds observation space


#################################################################
# Minicity Environment Instantiation Logic
#################################################################

class MinicityRouter(BaseRouter):
    """A router used to continuously re-route vehicles in minicity scenario.

    This class allows the vehicle to pick a random route at junctions.
    """

    def __init__(self, veh_id, router_params):
        self.prev_edge = None
        self.counter = 0 # Number of time steps that vehicle has not moved
        super().__init__(veh_id, router_params)

    def choose_route(self, env):
        """See parent class."""
        next_edge = None
        edge = env.vehicles.get_edge(self.veh_id) # modified from env.k.vehicle
        # if edge[0] == 'e_63':
        #     return ['e_63', 'e_94', 'e_52']
        subnetwork_edges = SUBROUTE_EDGES[SUBNETWORK.value]
        if edge not in subnetwork_edges:
            next_edge = None
        elif edge == self.prev_edge and self.counter < 5:
            next_edge = None
            self.counter += 1
        elif edge == self.prev_edge and self.counter >= 5:
            if type(subnetwork_edges[edge]) == str:
                next_edge = subnetwork_edges[edge]
            else:
                next_edge = random.choice(subnetwork_edges[edge])
            self.counter = 0
        elif type(subnetwork_edges[edge]) == str:
            next_edge = subnetwork_edges[edge]
            self.counter = 0
        elif type(subnetwork_edges[edge]) == list:
            next_edge = random.choice(subnetwork_edges[edge])
            self.counter = 0
        self.prev_edge = edge
        if next_edge is None:
            return None
        else:
            return [edge, next_edge]


def define_traffic_lights():
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

    return tl_logic


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
    sim_params = SumoParams(sim_step=0.25, emission_path='./data/')

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
        speed_mode="all_checks",
        lane_change_mode="strategic",
        initial_speed=0,
        num_vehicles=SUBNET_IDM[SUBNETWORK.value])
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(MinicityRouter, {}),
        # car_following_params=SumoCarFollowingParams(
        #     speed_mode="strategic",
        # ),
        speed_mode="all_checks",
        lane_change_mode="strategic",
        initial_speed=0,
        num_vehicles=SUBNET_RL[SUBNETWORK.value])
    
    additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
    additional_env_params['subnetwork'] = SUBNETWORK.value
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()

    if TRAFFIC_LIGHTS:
        additional_net_params['traffic_lights'] = True

    # Add inflows only on edges at border of subnetwork
    if len(SUBNET_INFLOWS[SUBNETWORK.value]) > 0:
        inflow = InFlows()
        for edge in SUBNET_INFLOWS[SUBNETWORK.value]:
            assert edge in SUBROUTE_EDGES[SUBNETWORK.value].keys()
            inflow.add(veh_type="idm",
                           edge=edge,
                           vehs_per_hour=1000, # Change this to modify bandwidth/traffic
                           departLane="free",
                           departSpeed=7.5)
            inflow.add(veh_type="rl",
                       edge=edge,
                       vehs_per_hour=1, # Change this to modify bandwidth/traffic
                       departLane="free",
                       departSpeed=7.5)
        net_params = NetParams(
            inflows=inflow,
            no_internal_links=False, additional_params=additional_net_params)
    else:
        net_params = NetParams(
                no_internal_links=False, additional_params=additional_net_params)


    initial_config = InitialConfig(
        spacing="random",
        min_gap=5,
        edges_distribution=list(SUBROUTE_EDGES[SUBNETWORK.value].keys())
    )

    if TRAFFIC_LIGHTS:
        scenario = MiniCityScenario(
            name="minicity",
            vehicles=vehicles,
            initial_config=initial_config,
            net_params=net_params,
            traffic_lights=define_traffic_lights())
    else:
        scenario = MiniCityScenario(
            name="minicity",
            vehicles=vehicles,
            initial_config=initial_config,
            net_params=net_params)


    if USE_CNN:
        #env = AccelCNNEnv(env_params, sim_params, scenario)
        env = AccelCNNSubnetEnv(env_params, sim_params, scenario)
    else:
        env = AccelEnv(env_params, sim_params, scenario)

    return SumoExperiment(env, scenario) # modified from Experiment(), added scenario param



if __name__ == "__main__":
    # import the experiment variable
    # There are six modes of pyglet rendering:
    # No rendering: minicity_example(render=False)
    # SUMO-GUI rendering: minicity_example(render=True)
    # Static grayscale rendering: minicity_example(render="gray")
    # Dynamic grayscale rendering: minicity_example(render="dgray")
    # Static RGB rendering: minicity_example(render="rgb")
    # Dynamic RGB rendering: minicity_example(render="drgb")

    # Change pxpm for Fangyu's renderer to fit within screen
    if RENDERER in ['gray', 'dgray', 'rgb', 'drgb']:
        pxpm = 1
    else:
        pxpm = 3

    exp = minicity_example(render=RENDERER,
                           save_render=False,
                           sight_radius=30,
                           pxpm=pxpm,
                           show_radius=True)

    # run for a set number of rollouts / time steps
    exp.run(1, 7200, convert_to_csv=True)
