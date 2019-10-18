from flow.envs import TestEnv
import numpy as np
from gym.spaces.box import Box

ADDITIONAL_ENV_PARAMS = {'target_nodes': [3369, 3341, 3370, 3344, 3329],
                         'num_incoming_edges_per_node': 4,
                         'num_stopbars': 3,
                         'num_advanced': 1,
                         'num_measures': 2}


class CoordinatedEnv(TestEnv):
    def __init__(self, env_params, sim_params, network, simulator='aimsun'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        self.additional_params = env_params.additional_params

        # veh_types = ["Car"]
        # self.k.vehicle.tracked_vehicle_types.update(veh_types)

        # target intersections
        self.target_nodes = env_params.additional_params["target_nodes"]
        self.edge_detector_dict = {}
        self.edges_with_detectors = {}
        for node_id in self.target_nodes:
            incoming_edges = self.k.traffic_light.get_incoming_edges(node_id)
            self.edge_detector_dict[node_id] = {}
            for edge_id in incoming_edges:
                detector_dict = self.k.traffic_light.get_detectors_on_edge(edge_id)
                stopbar = detector_dict['stopbar']
                advanced = detector_dict['advanced']
                type_map = {"stopbar": stopbar, "advanced": advanced}

                self.edge_detector_dict[node_id][edge_id] = type_map
                # for detector_id in stopbar:
                #     print(self.k.traffic_light.get_detector_flow_and_occupancy(detector_id), detector_id)
            # self.edges_with_detectors.update(incoming_edges)
        print(self.edge_detector_dict)

    @property
    def action_space(self):
        """See class definition."""
        cycle_length = 120
        return Box(
            low=-cycle_length,
            high=cycle_length,
            shape=(len(self.target_nodes), ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        ap = self.additional_params
        shape = len(self.target_nodes)*ap['num_incoming_edges_per_node']\
            * (ap['num_stopbars']+ap['num_advanced'])*ap['num_measures']
        return Box(low=0, high=1, shape=(shape, ), dtype=np.float32)

    # def get_state(self, rl_id=None, **kwargs):
    #     """See class definition."""
    #     # read detectors here
    #     ap = self.additional_params
    #     shape = len(self.target_nodes)*ap['num_incoming_edges_per_node']\
    #         * (ap['num_stopbars']+ap['num_advanced'])*ap['num_measures']
        
    #     the_state = np.zeros(shape)

    #     num_nodes = len(self.target_nodes)
    #     num_edges = ap['num_incoming_edges_per_node']
    #     num_detectors_types = (ap['num_stopbars']+ap['num_advanced'])
    #     num_measures = (ap['num_measures'])

    #     z = 0
    #     for i, (node, edge) in enumerate(self.edge_detector_dict.items()):
    #         for j, (edge_id, detector_info) in enumerate(edge.items()):
    #             for k, (detector_type, detector_ids) in enumerate(detector_info.items()):
    #                 if detector_type == 'stopbar':
    #                     for m, detector in enumerate(detector_ids):
    #                         flow, occupancy = self.k.traffic_light.get_detector_flow_and_occupancy(int(detector))
    #                         index = i*num_nodes + j*num_edges + k*num_detectors_types + m*num_measures
    #                         print(index)
    #                         the_state[index] = flow
    #                         the_state[index + 1] = occupancy
    #                         z+=2
    #                 elif detector_type == 'advanced':
    #                     flow, occupancy = 0, 0
    #                     for detector in detector_ids:
    #                         output = self.k.traffic_light.get_detector_flow_and_occupancy(int(detector))
    #                         flow += output[0]
    #                         occupancy += output[1]
    #                     index = i*num_nodes + j*num_edges + k*num_detectors_types + 3*num_measures
    #                     print(index)

    #                     the_state[index] = flow
    #                     the_state[index + 1] = occupancy
    #                     z+=2

    #                 # flow, occupancy = 0, 0
    #                 # for m, detector in enumerate(detector_ids):
    #                 #     if detector_type == 'stopbar':
    #                 #         flow, occupancy = self.k.traffic_light.get_detector_flow_and_occupancy(int(detector))
    #                 #         index = i*num_nodes + j*num_edges + k*num_detectors_types + m
    #                 #     elif detector_type == 'advanced':
    #                 #         flow, occupancy += self.k.traffic_light.get_detector_flow_and_occupancy(int(detector))
    #                 #         index = i*num_nodes + j*num_edges + k*num_detectors_types + 3
    #                 # the_state[index] = flow
    #                 # the_state[index + 1] = occupancy

    #                     # print(flow, occupancy)
    #     # print(self.k.simulation.time)
    #     print(z)
    #     return the_state

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""

        ap = self.additional_params

        num_nodes = len(self.target_nodes)
        num_edges = ap['num_incoming_edges_per_node']
        num_detectors_types = (ap['num_stopbars']+ap['num_advanced'])
        num_measures = (ap['num_measures'])

        shape = (num_nodes, num_edges, num_detectors_types, num_measures)
        the_state = np.zeros(shape) - 1

        for i, (node, edge) in enumerate(self.edge_detector_dict.items()):
            for j, (edge_id, detector_info) in enumerate(edge.items()):
                for k, (detector_type, detector_ids) in enumerate(detector_info.items()):
                    if detector_type == 'stopbar':
                        assert len(detector_ids) <= 3
                        for m, detector in enumerate(detector_ids):
                            flow, occupancy = self.k.traffic_light.get_detector_flow_and_occupancy(int(detector))
                            index = (i, j, m)
                            the_state[(*index, 0)] = flow
                            the_state[(*index, 1)] = occupancy
                    elif detector_type == 'advanced':
                        flow, occupancy = 0, 0
                        for detector in detector_ids:
                            output = self.k.traffic_light.get_detector_flow_and_occupancy(int(detector))
                            flow += output[0]
                            occupancy += output[1]
                        index = (i, j, ap['num_stopbars'])
                        the_state[(*index, 0)] = flow
                        the_state[(*index, 1)] = occupancy

        return the_state.flatten()

    def compute_reward(self, rl_actions, **kwargs):
        """Computes the average speed of vehicles in the network."""
        veh_ids = self.k.vehicle.get_ids()
        if len(veh_ids) == 0:
            return 0
        else:
            return sum(np.array(self.k.vehicle.get_speed(veh_ids)) < 0.05)/len(veh_ids)

    def additional_command(self):
        """Additional commands that may be performed by the step method."""
        pass
        # tl_ids = self.k.traffic_light.get_ids()
        # print(tl_ids)
        # print(self.k.traffic_light.set_intersection_offset(3344, -20))
        # print(self.k.traffic_light.get_detector_flow_and_occupancy(5157))
        # if self.k.simulation.time % 300 == 0:
        # print(self.get_state())
