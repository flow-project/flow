from flow.envs import Env
from flow.networks import Network
import numpy as np
from gym.spaces.box import Box

ADDITIONAL_ENV_PARAMS = {'target_nodes': [3369, 3341, 3370, 3344, 3329],
                         'num_incoming_edges_per_node': 4,
                         'num_stopbars': 3,
                         'num_advanced': 1,
                         'num_measures': 2,
                         'detection_interval': (0, 5, 0),
                         'statistical_interval': (0, 5, 0)}

np.random.seed(1234567890)


class CoordinatedEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='aimsun'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        self.additional_params = env_params.additional_params

        self.k.simulation.set_detection_interval(*self.additional_params['detection_interval'])

        self.k.simulation.set_statistical_interval(*self.additional_params['statistical_interval'])

        # veh_types = ["Car"]
        # self.k.vehicle.tracked_vehicle_types.update(veh_types)

        # target intersections
        self.target_nodes = env_params.additional_params["target_nodes"]
        self.current_offset = np.zeros(len(self.target_nodes))

        # reset_offsets
        for node_id in self.target_nodes:
            default_offset = self.k.traffic_light.get_intersection_offset(node_id)
            self.k.traffic_light.set_intersection_offset(node_id, -default_offset)

        self.edge_detector_dict = {}
        self.edges_with_detectors = {}
        # ap = self.additional_params
        # self.past_queues: = np.zeros(ap['num_incoming_edges_per_node']*len(self.target_nodes))
        self.past_queues = {}
        for node_id in self.target_nodes:
            incoming_edges = self.k.traffic_light.get_incoming_edges(node_id)
            self.edge_detector_dict[node_id] = {}
            for edge_id in incoming_edges:
                detector_dict = self.k.traffic_light.get_detectors_on_edge(edge_id)
                stopbar = detector_dict['stopbar']
                advanced = detector_dict['advanced']
                type_map = {"stopbar": stopbar, "advanced": advanced}

                self.edge_detector_dict[node_id][edge_id] = type_map
                self.past_queues[edge_id] = 0
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

    def _apply_rl_actions(self, rl_actions):
        for node_id, action in zip(self.target_nodes, rl_actions):
            self.k.traffic_light.set_intersection_offset(node_id, action)
        self.current_offset += np.array(rl_actions)

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""

        ap = self.additional_params

        num_nodes = len(self.target_nodes)
        num_edges = ap['num_incoming_edges_per_node']
        num_detectors_types = (ap['num_stopbars']+ap['num_advanced'])
        num_measures = (ap['num_measures'])

        shape = (num_nodes, num_edges, num_detectors_types, num_measures)
        the_state = np.zeros(shape)

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
                    if 7674941 in detector_ids:
                        print("%.2f\t%.2f\t%.2f" % (self.k.simulation.time, flow, occupancy))

        return the_state.flatten()

    def compute_reward(self, rl_actions, **kwargs):
        """Computes the average speed of vehicles in the network."""
        running_sum = 0
        for section_id, past_queue in self.past_queues.items():
            cumul_queue = self.k.traffic_light.get_cumulative_queue_length(section_id)
            queue = cumul_queue - self.past_queues[section_id]
            self.past_queues[section_id] = cumul_queue
            running_sum += queue**2
        return running_sum

    def additional_command(self):
        """Additional commands that may be performed by the step method."""
        # pass
        self.k.traffic_light.set_replication_seed(1)
        # str(self.k.traffic_light.get_ids())
        # tl_ids = self.k.traffic_light.get_ids()
        # print(tl_ids)
        # print(self.k.traffic_light.set_intersection_offset(3344, -20))
        # print(self.k.traffic_light.get_detector_flow_and_occupancy(5157))
        # if self.k.simulation.time % 300 == 0:
        # print(self.get_state())

    # def reset(self):
    #     replication_names = ['Replication 8050315', 'Replication 8050318',
    #                          'Replication 8050293', 'Replication (one hour)']


class CoordinatedNetwork(Network):
    pass
