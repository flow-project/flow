from flow.core import rewards

from flow.envs.base_env import Env

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np


class LoopMergesEnv(Env):
    """(description)

    States
    ------
    (blank)

    Actions
    -------
    (blank)

    RewardS
    -------
    (blank)

    Termination
    -----------
    (blank)

    Sorting
    -------
    (blank)
    """
    @property
    def action_space(self):
        return Box(low=-np.abs(self.env_params.max_decel),
                   high=self.env_params.max_accel,
                   shape=(self.vehicles.num_rl_vehicles,),
                   dtype=np.float32)

    @property
    def observation_space(self):
        speed = Box(low=0, high=np.inf, shape=(self.vehicles.num_vehicles,),
                    dtype=np.float32)
        pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,),
                  dtype=np.float32)
        edge = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,),
                   dtype=np.float32)
        return Tuple([speed, pos, edge])

    def _apply_rl_actions(self, rl_actions):
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]

        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        return rewards.desired_velocity(self, fail=kwargs["fail"])

    def get_state(self, **kwargs):
        sorted_pos = self.sorted_extra_data[0]
        edge_id = self.sorted_extra_data[1]

        # normalize everything
        for i in range(len(sorted_pos)):
            if edge_id[i] == 0:
                sorted_pos[i] = sorted_pos[i] / self.scenario.length
            elif edge_id[i] == 1:
                sorted_pos[i] = sorted_pos[i] / self.scenario.merge_in_len
            elif edge_id[i] == 2:
                sorted_pos[i] = sorted_pos[i] / self.scenario.merge_out_len

        return np.array([[self.vehicles.get_speed(veh_id),
                          sorted_pos[i],
                          edge_id[i]]
                         for i, veh_id in enumerate(self.sorted_ids)])

    def sort_by_position(self, **kwargs):
        """
        See parent class.

        Vehicles in the ring are sorted by their relative position in the ring,
        while vehicles outside the ring are sorted according to their position
        of their respective edge.

        Vehicles are sorted by position on the ring, then the in-merge, and
        finally the out-merge.
        """
        if self.scenario.merge_out_len is not None:
            pos = [[], [], []]
        else:
            pos = [[], []]

        for veh_id in self.vehicles.get_ids():
            this_edge = self.vehicles.get_edge(veh_id)
            this_pos = self.vehicles.get_position(veh_id)

            # the position of vehicles on the ring is their relative position
            # from the intersection with the merge-in
            if this_edge in ["ring_0", "ring_1"] or \
                    ":ring_0_%d" % self.scenario.lanes in this_edge or \
                    (":ring_1_%d" % self.scenario.lanes in this_edge and
                        self.scenario.merge_out_len is not None) or \
                    (":ring_1_0" in this_edge and
                        self.scenario.merge_out_len is None):
                pos[0].append(
                    (veh_id, self.vehicles.get_absolute_position(veh_id)
                     % self.scenario.length))

            # the position of vehicles in the merge-in / merge-out are their
            # relative position
            # from the start of the respective merge
            elif this_edge == "merge_in" or ":ring_0_0" in this_edge:
                if this_edge != "merge_in":
                    pos[1].append((veh_id, this_pos
                                   + self.scenario.merge_in_len))
                else:
                    pos[1].append((veh_id, this_pos))

            elif this_edge == "merge_out" or \
                    (":ring_1_0" in this_edge and
                        self.scenario.merge_out_len is not None):
                if this_edge == "merge_out":
                    pos[2].append((veh_id, this_pos
                                   + self.scenario.ring_1_0_len))
                else:
                    pos[2].append((veh_id, this_pos))

        sorted_ids = []
        sorted_pos = []
        for i in range(len(pos)):
            pos[i].sort(key=lambda tup: tup[1])
            if i == 0:
                sorted_ids += [tup[0] for tup in pos[i]]
                sorted_pos += [tup[1] % self.scenario.length for tup in pos[i]]
            else:
                sorted_ids += [tup[0] for tup in pos[i]]
                sorted_pos += [tup[1] for tup in pos[i]]

        edge_id = [0] * len(pos[0]) + [1] * len(pos[1])
        if self.scenario.merge_out_len is not None:
            edge_id += [2] * len(pos[2])

        # the extra data in this case is a tuple of sorted positions and
        # route ids
        sorted_extra_data = (sorted_pos, edge_id)

        return sorted_ids, sorted_extra_data

    def apply_acceleration(self, veh_ids, acc):
        """
        See parent class.

        In addition, merging vehicles travel at the target velocity at the
        merging lanes, and vehicle that are about to leave the network stop.
        """
        # index of vehicles in the list that have already received traci calls
        i_called = []

        for i, veh_id in enumerate(veh_ids):
            if "merge" in veh_id:
                this_edge = self.vehicles.get_edge(veh_id)
                this_pos = self.vehicles.get_position(veh_id)

                # vehicles that are about to exit are stopped
                if this_edge == "merge_out" and \
                        this_pos > self.scenario.merge_out_len - 10:
                    self.traci_connection.vehicle.slowDown(veh_id, 0, 1)
                    i_called.append(i)

                # vehicles in the merging lanes move at the target velocity
                # (if one is defined)
                elif "target_velocity" in self.env_params.additional_params \
                        and this_edge in ["merge_in", "merge_out"]:
                    self.traci_connection.vehicle.slowDown(
                        veh_id,
                        self.env_params.get_additional_param("target_velocity"),
                        duration=1
                    )
                    i_called.append(i)

        # delete data on vehicles that have already received traci calls in
        # order to reduce run-time
        veh_ids = [veh_ids[i] for i in range(len(veh_ids)) if i not in i_called]
        acc = [acc[i] for i in range(len(acc)) if i not in i_called]

        super().apply_acceleration(veh_ids=veh_ids, acc=acc)
