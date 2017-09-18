from cistar.envs.base_env import SumoEnvironment
from cistar.core import rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np
from numpy.random import normal


class TwoLoopsOneMergingEnvironment(SumoEnvironment):
    """
    Fully functional environment. Differs from the SimpleAccelerationEnvironment in loop_accel in that
    vehicles in this environment may follow one of two routes (continuously on the smaller ring or merging
    in and out of the smaller ring). Accordingly, the single global reference for position is replaced
    with a reference in each ring.
    """

    @property
    def action_space(self):
        """
        See parent class.
        Actions are a set of accelerations from max-deacc to max-acc for each rl vehicle.
        """
        max_acc = self.env_params.additional_params["max-acc"]
        max_deacc = - abs(self.env_params.additional_params["max-deacc"])

        return Box(low=-max_deacc, high=max_acc, shape=(self.vehicles.num_rl_vehicles, ))

    @property
    def observation_space(self):
        """
        See parent class.
        An observation is an array the velocities, positions, and edges for each vehicle
        """
        self.obs_var_labels = ["speed", "lane_pos", "edge_id"]
        speed = Box(low=0, high=np.inf, shape=(self.vehicles.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))
        edge_id = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))
        return Tuple((speed, absolute_pos, edge_id))

    def apply_rl_actions(self, rl_actions):
        """
        See parent class.
        """
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]
        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        return rewards.desired_velocity(self, fail=kwargs["fail"])

    def get_state(self, **kwargs):
        """
        See parent class.
        The state is an array the velocities, edge counts, and relative positions on the edge,
        for each vehicle.
        """
        vel = self.vehicles.get_speed(self.sorted_ids)
        pos = self.sorted_extra_data[0]
        edge = self.sorted_extra_data[1]
        max_speed = self.max_speed

        # divide the values by the maximum attainable speed
        normalized_vel = np.array(vel) / max_speed

        normalized_pos = []
        for i in range(len(pos)):
            if edge[i] == 0:
                # positions of vehicles in the ring are divided by the length of the ring
                normalized_pos.append(pos[i])
            elif edge[i] == 1:
                # positions of vehicles in the merge are divided by the length of the merge
                normalized_pos.append(pos[i])

        state = np.array([normalized_vel, normalized_pos, edge]).T
        return state

    def sort_by_position(self):
        """
        Instead of being sorted by a global reference, vehicles in this environment are sorted
        with regards to which ring this currently reside on.
        """
        sorted_ids = []
        sorted_edges = []
        sorted_pos = []

        veh_edges = self.vehicles.get_edge(self.ids)
        veh_pos = self.vehicles.get_position(self.ids)

        edge_list = [tup[0] for tup in self.scenario.total_edgestarts]
        edge_start_pos = [tup[1] for tup in self.scenario.total_edgestarts]

        for i, edge in enumerate(edge_list):
            veh_id_by_edge = [(self.ids[j], veh_pos[j] + edge_start_pos[i])
                              for j in range(len(self.ids)) if edge in veh_edges[j]]
            veh_id_by_edge.sort(key=lambda tup: tup[1])

            sorted_ids += [tup[0] for tup in veh_id_by_edge]
            # The edge ids of vehicles in the ring is set to 0, while those of vehicles outside
            # the ring are set to 1. In addition, the positions of vehicles in the ring are their
            # position on the ring starting from the left_top edge, while the positions of vehicles
            # on the merge is their position on the merge starting from the left_bottom edge.
            if i < 6:
                sorted_pos += [tup[1] for tup in veh_id_by_edge]
                sorted_edges += [0] * len(veh_id_by_edge)
            else:
                sorted_pos += [tup[1] - edge_start_pos[6] for tup in veh_id_by_edge]
                sorted_edges += [1] * len(veh_id_by_edge)

        return sorted_ids, (sorted_pos, sorted_edges)
