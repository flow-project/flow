import numpy as np
from gym.spaces import Box, Tuple, Discrete

from flow.envs import Env
from flow.networks import Network

ADDITIONAL_ENV_PARAMS = {'target_nodes': [3344],
                         # 'observed_nodes': [3386, 3371, 3362, 3373],
                         'num_incoming_edges_per_node': 4,
                         'num_stopbars': 3,
                         'num_advanced': 1,
                         'num_measures': 2,
                         'detection_interval': (0, 2, 0),
                         'statistical_interval': (0, 2, 0),
                         'replication_list': ['Replication 8050297',  # 5-11
                                              'Replication 8050315',  # 10-14
                                              'Replication 8050322']}  # 14-21
# the replication list should be copied in load.py

RLLIB_N_ROLLOUTS = 3  # copy from train_rllib.py

np.random.seed(1234567890)


class SingleLightEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='aimsun'):
        for param in ADDITIONAL_ENV_PARAMS:
            if param not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(param))

        super().__init__(env_params, sim_params, network, simulator)
        self.additional_params = env_params.additional_params

        self.episode_counter = 0
        self.detection_interval = self.additional_params['detection_interval'][1]*60  # assuming minutes for now
        self.k.simulation.set_detection_interval(*self.additional_params['detection_interval'])
        self.k.simulation.set_statistical_interval(*self.additional_params['statistical_interval'])
        self.k.traffic_light.set_replication_seed(np.random.randint(2e9))

        # target intersections
        self.target_nodes = env_params.additional_params["target_nodes"]
        self.total_green = 0

        # reset_phase_durations
        for node_id in self.target_nodes:
            default_offset = self.k.traffic_light.get_intersection_offset(node_id)
            self.k.traffic_light.change_intersection_offset(node_id, -default_offset)

        self.edge_detector_dict = {}
        self.edges_with_detectors = {}
        self.past_cumul_queue = {}
        self.observed_phases = []
        self.phases = []
        self.phase_array = []  # FOR CHECKING ONLY
        self.maxd_list = []

        # get cumulative queue lengths
        for node_id in self.target_nodes:
            self.node_id = node_id
            incoming_edges = self.k.traffic_light.get_incoming_edges(node_id)
            self.edge_detector_dict[node_id] = {}
            for edge_id in incoming_edges:
                detector_dict = self.k.traffic_light.get_detectors_on_edge(edge_id)
                stopbar = detector_dict['stopbar']
                advanced = detector_dict['advanced']
                type_map = {"stopbar": stopbar, "advanced": advanced}

                self.edge_detector_dict[node_id][edge_id] = type_map
                self.past_cumul_queue[edge_id] = 0

            # get control_id and # of rings
            self.control_id, self.num_rings = self.k.traffic_light.get_control_ids(node_id)
            # print(node_id, self.control_id, self.num_rings)

            for ring_id in range(0, self.num_rings):
                ring_phases = self.k.traffic_light.get_green_phases(node_id, ring_id)
                self.phases.append(ring_phases)  # get phases index per ring

            for phase_list in self.phases:
                for phase in phase_list:
                    self.observed_phases.append(phase)  # compile all green phases in a list
            print(self.observed_phases)

        self.current_phase_timings = np.zeros(int(len(self.observed_phases)/2))
        # reset phase duration
        for node_id in self.target_nodes:
            for phase in self.observed_phases:
                phase_duration, maxd, mind = self.k.traffic_light.get_duration_phase(node_id, phase)
                self.k.traffic_light.change_phase_duration(node_id, phase, phase_duration)
                print('initial phase: {} duration: {} max: {} min: {}'.format(phase, phase_duration, maxd, mind))
            self.total_green = self.k.traffic_light.get_total_green(node_id)
            print('cycle_length: {}'.format(self.total_green + 18))

        self.ignore_policy = False

    @property
    def action_space(self):
        """See class definition."""
        return Tuple(4 * (Discrete(40, ),))  # fixed for now, tempo fix

    @property
    def observation_space(self):
        """See class definition."""
        ap = self.additional_params
        shape = (len(self.target_nodes))*ap['num_incoming_edges_per_node']\
            * (ap['num_stopbars']+ap['num_advanced'])*ap['num_measures']
        return Box(low=0, high=5, shape=(shape, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        if self.ignore_policy:
            print('self.ignore_policy is True')
            return
        actions = np.array(rl_actions).flatten()
        self.phase_array = []
        self.maxd_list = []
        for phase_list in self.phases:
            for phase, action in zip(phase_list, actions):
                if action:
                    self.k.traffic_light.change_phase_duration(self.node_id, phase, action)
                    # print(phase, action)
                    phase_duration, maxd, _ = self.k.traffic_light.get_duration_phase(self.node_id, phase)
                    self.phase_array.append(phase_duration)
                    self.maxd_list.append(maxd)

        self.current_phase_timings = actions

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""

        ap = self.additional_params

        num_nodes = len(self.target_nodes)
        num_edges = ap['num_incoming_edges_per_node']
        num_detectors_types = (ap['num_stopbars']+ap['num_advanced'])
        num_measures = (ap['num_measures'])
        normal = 2000

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
                            state[(*index, 0)] = (count/self.detection_interval)/(normal/3600)
                            state[(*index, 1)] = occupancy
                    elif detector_type == 'advanced':
                        flow, occupancy = 0, 0
                        for detector in detector_ids:
                            count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                            flow += (count/self.detection_interval)/(normal/3600)
                            occupancy += occupancy
                        index = (i, j, ap['num_stopbars'])
                        state[(*index, 0)] = flow
                        state[(*index, 1)] = occupancy

        return state.flatten()

    def compute_reward(self, rl_actions, **kwargs):
        """Computes the sum of queue lengths at all intersections in the network."""
        reward = 0
        for section_id in self.past_cumul_queue:
            current_cumul_queue = self.k.traffic_light.get_cumulative_queue_length(section_id)
            queue = current_cumul_queue - self.past_cumul_queue[section_id]
            self.past_cumul_queue[section_id] = current_cumul_queue

            # reward is negative queues
            reward -= (queue**2) * 100

        print(f'{self.k.simulation.time:.0f}', '\t', f'{reward:.2f}', '\t',
              self.current_phase_timings.flatten(), '\t', sum(self.current_phase_timings.flatten()))
        # print(self.phase_array)
        # print(self.maxd_list)

        return reward

    def step(self, rl_actions):
        """See parent class."""

        self.time_counter += self.env_params.sims_per_step
        self.step_counter += self.env_params.sims_per_step

        self.apply_rl_actions(rl_actions)

        # advance the simulation in the simulator by one step
        self.k.simulation.simulation_step()

        for _ in range(self.env_params.sims_per_step):
            self.k.simulation.update(reset=False)

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        done = (self.time_counter >= self.env_params.warmup_steps +
                self.env_params.horizon)  # or crash

        # compute the info for each agent
        infos = {}

        # get control_id for every step
        self.control_id, self.num_rings = self.k.traffic_light.get_control_ids(self.node_id)

        # compute the reward
        reward = self.compute_reward(rl_actions)

        return next_observation, reward, done, infos

    def reset(self):
        """See parent class.

        The AIMSUN simulation is reset along with other variables.
        """
        # reset the step counter
        self.step_counter = 0

        if self.episode_counter:
            self.k.simulation.reset_simulation()

            episode = self.episode_counter % RLLIB_N_ROLLOUTS

            print('-----------------------')
            print(f'Episode {RLLIB_N_ROLLOUTS if not episode else episode} of {RLLIB_N_ROLLOUTS} complete')
            print('Resetting simulation')
            print('-----------------------')

        # increment episode count
        self.episode_counter += 1

        # reset variables
        self.current_phase_timings = np.zeros(int(len(self.observed_phases)/2))
        for section_id in self.past_cumul_queue:
            self.past_cumul_queue[section_id] = 0

        # perform the generic reset function
        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        return observation


class CoordinatedNetwork(Network):
    pass
