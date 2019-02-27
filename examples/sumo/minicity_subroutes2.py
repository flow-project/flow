"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.controllers import BaseRouter
from flow.core.experiment import SumoExperiment # Modified from Experiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, InFlows
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
#from flow.core.params import VehicleParams
from flow.core.vehicles import Vehicles # Modified from VehicleParams
from flow.envs.loop.loop_accel import * #AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario, ADDITIONAL_NET_PARAMS
from flow.core.traffic_lights import TrafficLights
import numpy as np
import random
from enum import Enum

from matplotlib import pyplot as plt

np.random.seed(204)


# Definitions of subnetworks
class SubRoute(Enum):
    ALL =  0
    SUB1 = 1 # top left with merge
    SUB2 = 2 # top center intersection
    SUB3 = 3 # top right intersection
    SUB4 = 4 # center intersection
    SUB5 = 5 # bottom left
    SUB6 = 6 # bottom right

    TOP_LEFT = 7 #previous breakdown
    TOP_RIGHT = 8 #previous breakdown
    BOTTOM = 9 #previous breakdown
    FULL_RIGHT = 10 # Aboudy's


SUBNETWORK = SubRoute.ALL  # CHANGE THIS PARAMETER TO SELECT CURRENT SUBNETWORK

                            # Set it to SubRoute.ALL, SubRoute.TOP_LEFT, etc.

TRAFFIC_LIGHTS = True       # CHANGE THIS to True to add traffic lights to Minicity

RENDERER = True   #'drgb'           # PARAMETER. 
                            # Set to True to use default Sumo renderer, 
                            # Set to 'drgb' for Fangyu's renderer



# Denotes the route choice behavior of vehicles on an edge.
#
# The key is the name of the starting edge of the vehicle, and the element if
# the name of the next edge the vehicle will take.
#
# 1. If the element is a string name of an edge, the vehicle will always take
#    that edge as a next edge.
# 2. If the element is a list of strings, the vehicle will uniformly choose
#    among all written edges.
# 3. If the element is a list of tuples, the specific next edges are chosen
#    using the weighting specific by the integer second element.
SUBROUTE_EDGES = [
    # top left with merge
    {

    }

    # top center intersection
    {

    }

    # top right intersection
    {

    }

    # center intersection
    {

    }

    # bottom left
    {

    }

    # bottom right
    {

    }






    # Full network
    {
    'e_12': ['e_18', 'e_13'],
    'e_18': 'e_19',
    'e_19': 'e_24',
    'e_24': 'e_33',
    'e_33': ['e_45', 'e_46', 'e_49'],
    'e_13': 'e_14',
    'e_14': ['e_22', 'e_15'],
    'e_22': 'e_33',
    'e_15': 'e_16',
    # 'e_16': 'e_20',
    'e_20': ['e_47', 'e_48'],
    'e_47': ['e_34', 'e_45', 'e_49'],
    'e_45': 'e_43',
    'e_43': 'e_41',
    'e_41': ['e_88','e_39', 'e_50'],
    'e_88': 'e_26',
    'e_26': ['e_12', 'e_2'],
    'e_34': 'e_23',
    'e_23': ['e_15', 'e_5'],
    'e_5': 'e_4',
    'e_4': 'e_3',
    'e_3': ['e_25', 'e_2'],
    'e_87': ['e_40', 'e_50', 'e_39'],
    'e_40': 'e_42',
    'e_42': 'e_44',
    'e_44': ['e_34', 'e_46'],
    'e_46': 'e_35',
    'e_35': 'e_27',
    'e_27': 'e_6',
    'e_6': ['e_22', 'e_5'],

    'e_48': ['e_78', 'e_81'],
    'e_78': ['e_86', 'e_76'],
    'e_86': ['e_55', 'e_59'],
    'e_56': 'e_89',
    'e_89': ['e_74', 'e_80', 'e_75'],
    'e_80': 'e_83',
    'e_83': 'e_82',
    'e_82': ['e_79', 'e_78'],
    'e_59': ['e_46', 'e_45', 'e_34'],
    'e_76': ['e_90', 'e_74', 'e_80'],
    'e_74': ['e_70', 'e_72'],
    'e_70': 'e_61',
    'e_61': 'e_54',
    'e_54': ['e_40', 'e_88', 'e_39'],
    'e_50': 'e_60',
    'e_60': 'e_69',
    'e_69': ['e_73', 'e_72'],
    'e_73': ['e_80', 'e_75', 'e_90'],
    'e_90': 'e_62',
    'e_62': 'e_57',
    'e_57': ['e_58', 'e_59'],
    'e_58': ['e_76', 'e_77'],
    'e_75': ['e_77', 'e_86'],
    'e_77': ['e_79', 'e_81'],
    'e_81': 'e_84',
    'e_84': 'e_85',
    'e_85': ['e_75', 'e_90', 'e_74'],
    'e_79': ['e_47', 'e_35'],
    'e_49': ['e_55', 'e_58'],
    'e_55': 'e_56',

    'e_25': ['e_87', 'e_30'],
    'e_30': 'e_31',
    'e_31': 'e_32',
    'e_32': 'e_21',
    'e_39': 'e_37',
    'e_37': 'e_29_u',
    'e_29_u': 'e_21',
    'e_21': 'e_8_u',
    'e_9': ['e_10', 'e_92'],
    'e_92': 'e_7',
    'e_7': ['e_8_b', 'e_17'],
    'e_8_b': 'e_8_u',
    'e_17': 'e_28_b',
    # 'e_28_b': 'e_36',
    'e_36': 'e_93',
    'e_93': 'e_53',
    'e_53': 'e_64',
    'e_64': ['e_65', 'e_67'],
    'e_65': 'e_66',
    'e_66': ['e_91', 'e_63'],
    'e_63': 'e_94',
    'e_94': ['e_51', 'e_52'],
    'e_51': 'e_29_u',
    'e_52': 'e_38',
    'e_38': ['e_50', 'e_88', 'e_40'],
    'e_72': 'e_68',
    'e_68': 'e_66',
    'e_67': 'e_71',
    'e_71': ['e_70', 'e_73'],
    'e_8_u': 'e_9',
    'e_10': 'e_11',
    'e_11': ['e_25', 'e_12'],
    'e_2': 'e_1',
    'e_1': 'e_7'
    },

    # Top left
    {
    'e_12': ['e_18', 'e_13'],
    'e_18': 'e_19',
    'e_19': 'e_24',
    'e_24': 'e_33',
    'e_33': ['e_45', 'e_46'],
    'e_13': 'e_14',
    'e_14': ['e_22', 'e_15'],
    'e_22': 'e_33',
    'e_15': 'e_16',
    # 'e_16': 'e_20',
    'e_20': 'e_47',
    'e_47': ['e_34', 'e_45'],
    'e_45': 'e_43',
    'e_43': 'e_41',
    'e_41': 'e_88',
    'e_88': 'e_26',
    'e_26': 'e_12',
    'e_34': 'e_23',
    'e_23': ['e_15', 'e_5'],
    'e_5': 'e_4',
    'e_4': 'e_3',
    'e_3': 'e_25',
    'e_25': 'e_87',
    'e_87': 'e_40',
    'e_40': 'e_42',
    'e_42': 'e_44',
    'e_44': ['e_34', 'e_46'],
    'e_46': 'e_35',
    'e_35': 'e_27',
    'e_27': 'e_6',
    'e_6': ['e_22', 'e_5']
    },

    # Top right
    {
    # 'e_40': 'e_42',
    'e_42': 'e_44',
    'e_44': ['e_49', 'e_46'],
    'e_46': 'e_48',
    'e_48': ['e_78', 'e_81'],
    'e_78': ['e_86', 'e_76'],
    'e_86': ['e_55', 'e_59'],
    'e_56': 'e_89',
    'e_89': ['e_74', 'e_80', 'e_75'],
    'e_80': 'e_83',
    'e_83': 'e_82',
    'e_82': ['e_79', 'e_78'],
    'e_59': ['e_46', 'e_45'],
    'e_76': ['e_90', 'e_74', 'e_80'],
    'e_74': 'e_70',
    'e_70': 'e_61',
    'e_61': 'e_54',
    'e_54': 'e_40',
    'e_45': 'e_43',
    'e_43': 'e_41',
    'e_41': 'e_50',
    'e_50': 'e_60',
    'e_60': 'e_69',
    'e_69': 'e_73',
    'e_73': ['e_80', 'e_90'], #['e_80', 'e_75', 'e_90'],
    'e_90': 'e_62',
    'e_62': 'e_57',
    'e_57': ['e_58', 'e_59'],
    'e_58': ['e_76', 'e_77'],
    'e_75': ['e_77', 'e_86'],
    'e_77': ['e_79', 'e_81'],
    'e_81': 'e_84',
    'e_84': 'e_85',
    'e_85': ['e_75', 'e_90', 'e_74'],
    'e_79': 'e_47',
    'e_47': 'e_45',
    'e_49': ['e_55', 'e_58'],
    'e_55': 'e_56' 
    },

    # Bottom
    {
    'e_25': ['e_87', 'e_30'],
    'e_30': 'e_31',
    'e_31': 'e_32',
    'e_32': 'e_21',
    'e_87': ['e_39', 'e_50'],
    'e_39': 'e_37',
    'e_37': 'e_29_u',
    'e_29_u': 'e_21',
    'e_21': 'e_8_u',
    'e_9': ['e_10', 'e_92'],
    'e_92': 'e_7',
    'e_7': ['e_8_b', 'e_17'],
    'e_8_b': 'e_8_u',
    'e_17': 'e_28_b',
    # 'e_28_b': 'e_36',
    'e_36': 'e_93',
    'e_93': 'e_53',
    'e_53': 'e_64',
    'e_64': ['e_65', 'e_67'],
    'e_65': 'e_66',
    'e_66': ['e_91', 'e_63'],
    'e_63': 'e_94',
    'e_94': ['e_51', 'e_52'],
    'e_51': 'e_29_u',
    'e_52': 'e_38',
    'e_38': ['e_50', 'e_88'],
    'e_50': 'e_60',
    'e_60': 'e_69',
    'e_69': 'e_72',
    'e_72': 'e_68',
    'e_68': 'e_66',
    'e_67': 'e_71',
    'e_71': 'e_70',
    'e_70': 'e_61',
    'e_61': 'e_54',
    'e_54': ['e_88', 'e_39'],
    'e_8_u': 'e_9',
    'e_10': 'e_11',
    'e_11': 'e_25',
    # 'e_88': 'e_26',
    'e_26': 'e_2',
    'e_2': 'e_1',
    'e_1': 'e_7'
    },

    # Full right (Aboudy's)
    {
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
    },
]

# The cropping dimensions for a subnetwork out of whole Minicity.
# Contains (minWidth, maxWidth, minHeight, maxHeight) 

SUBNET_CROP = [
    (0, 5000, 0, 5000), # Full network
    (0, 5000, 0, 5000) #top left with merge
    (0, 5000, 0, 5000) # top center intersection
    (0, 5000, 0, 5000) # top right intersection
    (0, 5000, 0, 5000) # center intersection
    (0, 5000, 0, 5000) # bottom left
    (0, 5000, 0, 5000) # bottom right 

    (0, 920, 0, 1020),  # Top left
    (890, 5000, 0, 1020), # Top right
    (0, 3000, 970, 5000), # Bottom
    (2500, 5000, 0, 5000), # Full right
]

# Whether pre-defined subnetwork is not a self-contained loop.
# If routes are clipped and vehicles can exit subnetwork, requires vehicle inflows
REQUIRES_INFLOWS = [
    False, # Full network
    True, #top left with merge
    True, #top center intersection
    True, #top right intersection
    True, #center intersection
    True, #bottom left
    True, #bottom right

    True, # Top-left
    True, # Top-right
    True, # Bottom
    True,  # Full-right (Aboudy's)
]


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
        # if edge[0] == 'e_63':
        #     return ['e_63', 'e_94', 'e_52']
        subnetwork_edges = SUBROUTE_EDGES[SUBNETWORK.value]
        if edge not in subnetwork_edges or edge == self.prev_edge:
            next_edge = None
        elif type(subnetwork_edges[edge]) == str:
            next_edge = subnetwork_edges[edge]
        elif type(subnetwork_edges[edge]) == list:
            if type(subnetwork_edges[edge][0]) == str:
                next_edge = random.choice(subnetwork_edges[edge])
            else:
                # Edge choices weighted by integer. 
                # Inefficient untested implementation, but doesn't rely on numpy.random.choice or Python >=3.6 random.choices 
                next_edge = random.choice(sum(([edge]*weight for edge, weight in subnetwork_edges), []))
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
        speed_mode="no_collide",
        lane_change_mode="strategic",
        initial_speed=0,
        num_vehicles=50)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(MinicityRouter, {}),
        # car_following_params=SumoCarFollowingParams(
        #     speed_mode="strategic",
        # ),
        speed_mode="no_collide",
        lane_change_mode="strategic",
        initial_speed=0,
        num_vehicles=0)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()

    if TRAFFIC_LIGHTS:
        additional_net_params['traffic_lights'] = True

    if REQUIRES_INFLOWS[SUBNETWORK.value]:
        # Add vehicle inflows to account for clipped subnetworks
        inflow = InFlows()
        for edge in SUBROUTE_EDGES[SUBNETWORK.value].keys():
            inflow.add(veh_type="idm",
                       edge=edge,
                       vehs_per_hour=5, # Change this to modify bandwidth/traffic
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

    #env = AccelEnv(env_params, sim_params, scenario)
    env = AccelSubnetEnv(env_params, sim_params, scenario)
    #env = AccelCNNSubnetEnv(env_params, sim_params, scenario)
    #env = AccelCNNEnv(env_params, sim_params, scenario)

    return SumoExperiment(env, scenario) # modified from Experiment(), added scenario param

class AccelSubnetEnv(AccelEnv):

    def render(self, reset=False, buffer_length=5):
        """Render a frame.
        Parameters
        ----------
        reset: bool
            set to True to reset the buffer
        buffer_length: int
            length of the buffer
        """
        if self.sumo_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            # render a frame
            self.pyglet_render()

            # cache rendering
            if reset:
                self.frame_buffer = [self.frame.copy() for _ in range(5)]
                self.sights_buffer = [self.sights.copy() for _ in range(5)]

                # Crop self.frame_buffer to subnetwork only
                for frame in self.frame_buffer:
                    subnet_xmin = SUBNET_CROP[SUBNETWORK.value][0]
                    subnet_xmax = SUBNET_CROP[SUBNETWORK.value][1]
                    subnet_ymin = SUBNET_CROP[SUBNETWORK.value][2]
                    subnet_ymax = SUBNET_CROP[SUBNETWORK.value][3]
                    frame = frame[subnet_ymin:subnet_ymax, \
                                 subnet_xmin:subnet_xmax, :]
            else:
                if self.step_counter % int(1/self.sim_step) == 0:
                    next_frame = self.frame.copy()
                    subnet_xmin = SUBNET_CROP[SUBNETWORK.value][0]
                    subnet_xmax = SUBNET_CROP[SUBNETWORK.value][1]
                    subnet_ymin = SUBNET_CROP[SUBNETWORK.value][2]
                    subnet_ymax = SUBNET_CROP[SUBNETWORK.value][3]
                    next_frame = next_frame[subnet_ymin:subnet_ymax, \
                                 subnet_xmin:subnet_xmax, :]

                    # Save a cropped image to current executing directory for debug
                    # plt.imsave('test_subnet_crop.png', next_frame)


                    self.frame_buffer.append(next_frame)
                    self.sights_buffer.append(self.sights.copy())

                if len(self.frame_buffer) > buffer_length:
                    self.frame_buffer.pop(0)
                    self.sights_buffer.pop(0)


class AccelCNNSubnetEnv(AccelCNNEnv):

    # Currently has a bug with "sights_buffer / 255" in original AccelCNNEnv
    pass

    # def render(self, reset=False, buffer_length=5):
    #     """Render a frame.
    #     Parameters
    #     ----------
    #     reset: bool
    #         set to True to reset the buffer
    #     buffer_length: int
    #         length of the buffer
    #     """
    #     if self.sumo_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
    #         # render a frame
    #         self.pyglet_render()

    #         # cache rendering
    #         if reset:
    #             self.frame_buffer = [self.frame.copy() for _ in range(5)]
    #             self.sights_buffer = [self.sights.copy() for _ in range(5)]

    #             # Crop self.frame_buffer to subnetwork only
    #             for frame in self.frame_buffer:
    #                 subnet_xmin = SUBNET_CROP[SUBNETWORK.value][0]
    #                 subnet_xmax = SUBNET_CROP[SUBNETWORK.value][1]
    #                 subnet_ymin = SUBNET_CROP[SUBNETWORK.value][2]
    #                 subnet_ymax = SUBNET_CROP[SUBNETWORK.value][3]
    #                 frame = frame[subnet_ymin:subnet_ymax, \
    #                              subnet_xmin:subnet_xmax, :]
    #         else:
    #             if self.step_counter % int(1/self.sim_step) == 0:
    #                 next_frame = self.frame.copy()
    #                 subnet_xmin = SUBNET_CROP[SUBNETWORK.value][0]
    #                 subnet_xmax = SUBNET_CROP[SUBNETWORK.value][1]
    #                 subnet_ymin = SUBNET_CROP[SUBNETWORK.value][2]
    #                 subnet_ymax = SUBNET_CROP[SUBNETWORK.value][3]
    #                 next_frame = next_frame[subnet_ymin:subnet_ymax, \
    #                              subnet_xmin:subnet_xmax, :]

    #                 # Save a cropped image to current executing directory for debug
    #                 # plt.imsave('test_subnet_crop.png', next_frame)


    #                 self.frame_buffer.append(next_frame)
    #                 self.sights_buffer.append(self.sights.copy())

    #             if len(self.frame_buffer) > buffer_length:
    #                 self.frame_buffer.pop(0)
    #                 self.sights_buffer.pop(0)


if __name__ == "__main__":
    # import the experiment variable
    # There are six modes of pyglet rendering:
    # No rendering: minicity_example(render=False)
    # SUMO-GUI rendering: minicity_example(render=True)
    # Static grayscale rendering: minicity_example(render="gray")
    # Dynamic grayscale rendering: minicity_example(render="dgray")
    # Static RGB rendering: minicity_example(render="rgb")
    # Dynamic RGB rendering: minicity_example(render="drgb")
    exp = minicity_example(render=RENDERER,
                           save_render=False,
                           sight_radius=30,
                           pxpm=3,
                           show_radius=True)

    # run for a set number of rollouts / time steps
    exp.run(1, 7200, convert_to_csv=True)
