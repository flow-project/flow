import numpy as np
from gym.spaces.box import Box

from flow.envs import Env
from flow.networks import Network

ADDITIONAL_ENV_PARAMS = {'target_nodes': [3369, 3341, 3370, 3344, 3329],
                         'num_incoming_edges_per_node': 4,
                         'num_stopbars': 3,
                         'num_advanced': 1,
                         'num_measures': 2,
                         'detection_interval': (0, 5, 0),
                         'statistical_interval': (0, 5, 0),
                         'replication_list': ['Replication 8050297',  # 5-11
                                              'Replication 8050315',  # 10-14
                                              'Replication 8050322']}  # 14-21
# the replication list should be copied in load.py

RLLIB_N_ROLLOUTS = 6  # copied from train_rllib.py

np.random.seed(1234567890)


class CoordinatedEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='aimsun'):
        for param in ADDITIONAL_ENV_PARAMS:
            if param not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(param))

        super().__init__(env_params, sim_params, network, simulator)
        self.additional_params = env_params.additional_params

        self.episode_count = 0
        self.detection_interval = self.additional_params['detection_interval'][1]*60  # assuming minutes for now
        self.k.simulation.set_detection_interval(*self.additional_params['detection_interval'])
        self.k.simulation.set_statistical_interval(*self.additional_params['statistical_interval'])
        self.k.traffic_light.set_replication_seed(np.random.randint(2e9))

        # target intersections
        self.target_nodes = env_params.additional_params["target_nodes"]
        self.current_offset = np.zeros(len(self.target_nodes))

        # reset_offsets
        for node_id in self.target_nodes:
            default_offset = self.k.traffic_light.get_intersection_offset(node_id)
            self.k.traffic_light.change_intersection_offset(node_id, -default_offset)

        self.edge_detector_dict = {}
        self.edges_with_detectors = {}
        self.past_cumul_queue = {}
        for node_id in self.target_nodes:
            incoming_edges = self.k.traffic_light.get_incoming_edges(node_id)
            self.edge_detector_dict[node_id] = {}
            for edge_id in incoming_edges:
                detector_dict = self.k.traffic_light.get_detectors_on_edge(edge_id)
                stopbar = detector_dict['stopbar']
                advanced = detector_dict['advanced']
                type_map = {"stopbar": stopbar, "advanced": advanced}

                self.edge_detector_dict[node_id][edge_id] = type_map
                self.past_cumul_queue[edge_id] = 0

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=0,
            high=1,
            shape=(len(self.target_nodes),),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        ap = self.additional_params
        shape = len(self.target_nodes)*ap['num_incoming_edges_per_node']\
            * (ap['num_stopbars']+ap['num_advanced'])*ap['num_measures']
        return Box(low=0, high=5, shape=(shape, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        actions = rl_actions * 120
        delta_offset = actions - self.current_offset
        for node_id, action in zip(self.target_nodes, delta_offset):
            self.k.traffic_light.change_intersection_offset(node_id, action)
        self.current_offset = actions

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""

        ap = self.additional_params

        num_nodes = len(self.target_nodes)
        num_edges = ap['num_incoming_edges_per_node']
        num_detectors_types = (ap['num_stopbars']+ap['num_advanced'])
        num_measures = (ap['num_measures'])

        shape = (num_nodes, num_edges, num_detectors_types, num_measures)
        state = np.zeros(shape)

        for i, (node, edge) in enumerate(self.edge_detector_dict.items()):
            for j, (edge_id, detector_info) in enumerate(edge.items()):
                for detector_type, detector_ids in detector_info.items():
                    if detector_type == 'stopbar':
                        assert len(detector_ids) <= 3
                        for m, detector in enumerate(detector_ids):
                            count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                            index = (i, j, m)
                            state[(*index, 0)] = (count/self.detection_interval)/(2000/3600)
                            state[(*index, 1)] = occupancy
                    elif detector_type == 'advanced':
                        flow, occupancy = 0, 0
                        for detector in detector_ids:
                            count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                            flow += (count/self.detection_interval)/(2000/3600)
                            occupancy += occupancy
                        index = (i, j, ap['num_stopbars'])
                        state[(*index, 0)] = flow
                        state[(*index, 1)] = occupancy
                    if 7674941 in detector_ids:
                        print("%.2f\t%.2f\t%.2f" % (self.k.simulation.time, flow, occupancy))

        return state.flatten()

    def compute_reward(self, rl_actions, **kwargs):
        """Computes the sum of queue lengths at all intersections in the network."""
        running_sum = 0
        for section_id in self.past_cumul_queue:
            current_cumul_queue = self.k.traffic_light.get_cumulative_queue_length(section_id)
            delta_queue = current_cumul_queue - self.past_cumul_queue[section_id]
            self.past_cumul_queue[section_id] = current_cumul_queue
            running_sum += np.copysign(delta_queue**2, delta_queue)
        print(self.current_offset, -running_sum)

        # reward is negative queues
        return -running_sum

    def reset(self):
        """See parent class.

        The AIMSUN simulation is reset along with other variables.
        """
        # reset the step counter
        self.step_counter = 0

        if self.episode_count:
            self.k.simulation.reset_simulation()
            print('-----------------------')
            print(f'Episode {self.episode_count % RLLIB_N_ROLLOUTS} of {RLLIB_N_ROLLOUTS} complete')
            print('Resetting AIMSUN')
            print('-----------------------')

        # perform the generic reset function
        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        # increment episode count
        self.episode_count += 1

        # reset variables
        self.current_offset = np.zeros(len(self.target_nodes))
        for section_id in self.past_cumul_queue:
            self.past_cumul_queue[section_id] = 0

        return observation


class CoordinatedNetwork(Network):
    pass
