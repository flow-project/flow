from cistar.envs.loop import LoopEnvironment
from cistar.core import rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np
from numpy.random import normal


class SimpleLoopMergesEnvironment(LoopEnvironment):
    """
    Fully functional environment. Takes in an *acceleration* as an action. Reward function is negative norm of the
    difference between the velocities of each vehicle, and the target velocity. State function is a vector of the
    velocities for each vehicle.
    """

    @property
    def action_space(self):
        """
        Actions are a set of accelerations from 0 to 15m/s
        :return:
        """
        return Box(low=-np.abs(self.env_params.get_additional_param("max-deacc")), high=self.env_params.get_additional_param("max-acc"),
                   shape=(self.scenario.num_rl_vehicles,))

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        speed = Box(low=0, high=np.inf, shape=(self.scenario.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        edge = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        return Tuple([speed, absolute_pos, edge])

    def apply_rl_actions(self, rl_actions):
        """
        See parent class
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
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities, relative positions, and edge ids for each vehicle
        """
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

        return np.array([[self.vehicles[veh_id]["speed"] + normal(0, self.observation_vel_std),
                          sorted_pos[i] + normal(0, self.observation_pos_std),
                          edge_id[i]] for i, veh_id in enumerate(self.sorted_ids)])

    def additional_command(self):
        """
        Vehicles that are meant to stay in the ring are rerouted whenever they reach a new edge.
        """
        for veh_id in self.ids:
            # if the vehicle is one the merging vehicles, and there is a merge-out lane, it should not be rerouted
            if "merge" in self.vehicles[veh_id]["type"] and self.scenario.net_params["merge_out_length"] is not None:
                continue

            # check if a vehicle needs to be rerouted
            route = None
            if self.vehicles[veh_id]["route"][-1] == self.vehicles[veh_id]["edge"]:
                if self.vehicles[veh_id]["edge"] == "ring_0":
                    route = ["ring_0", "ring_1"]
                elif self.vehicles[veh_id]["edge"] == "ring_1":
                    route = ["ring_1", "ring_0"]

            # perform rerouting and update vehicle's perception of its route
            if route is not None:
                self.vehicles[veh_id]["route"] = route
                self.traci_connection.vehicle.setRoute(vehID=veh_id, edgeList=route)

    def sort_by_position(self, **kwargs):
        """
        See parent class
        Vehicles in the ring are sorted by their relative position in the ring, while vehicles outside the ring
        are sorted according to their position of their respective edge.
        Vehicles are sorted by position on the ring, then the in-merge, and finally the out-merge.
        """
        if self.scenario.merge_out_len is not None:
            pos = [[], [], []]
        else:
            pos = [[], []]

        for veh_id in self.ids:
            this_edge = self.vehicles[veh_id]["edge"]
            this_pos = self.vehicles[veh_id]["position"]

            # the position of vehicles on the ring is their relative position from the
            # intersection with the merge-in
            if this_edge in ["ring_0", "ring_1"] or \
                    ":ring_0_%d" % self.scenario.lanes in this_edge or \
                    (":ring_1_%d" % self.scenario.lanes in this_edge and self.scenario.merge_out_len is not None) or \
                    (":ring_1_0" in this_edge and self.scenario.merge_out_len is None):
                # # merging vehicles need to update their absolute position once to adjust for the number of loops
                # # non-merging vehicles have performs
                # if "merge" in veh_id and \
                #                 self.vehicles[self.vehicles[veh_id]["follower"]]["absolute_position"] > \
                #                 self.vehicles[veh_id]["absolute_position"]:
                #     lag_id = self.vehicles[veh_id]["follower"]
                #
                #     # number of loops performed by the lagging vehicle
                #     num_loops = int(self.vehicles[lag_id]["absolute_position"] / self.scenario.length)
                #
                #     # TODO: might not want to change it completely, but instead calculate in each iteration
                #     self.vehicles[veh_id]["absolute_position"] += (num_loops + 1) * self.scenario.length
                #
                # pos[0].append((veh_id, self.vehicles[veh_id]["absolute_position"]))
                pos[0].append((veh_id, self.vehicles[veh_id]["absolute_position"] % self.scenario.length))

            # the position of vehicles in the merge-in / merge-out are their relative position
            # from the start of the respective merge
            elif this_edge == "merge_in" or ":ring_0_0" in this_edge:
                if this_edge != "merge_in":
                    pos[1].append((veh_id, this_pos + self.scenario.merge_in_len))
                else:
                    pos[1].append((veh_id, this_pos))

            elif this_edge == "merge_out" or (":ring_1_0" in this_edge and self.scenario.merge_out_len is not None):
                if this_edge == "merge_out":
                    pos[2].append((veh_id, this_pos + self.scenario.ring_1_0_len))
                else:
                    pos[2].append((veh_id, this_pos))

        sorted_ids = []
        sorted_pos = []
        for i in range(len(pos)):
            pos[i].sort(key=lambda tup: tup[1])
            if i == 0:
                if len(self.rl_ids) == 1:
                    # for single rl vehicle case: set the rl vehicle in index 0 to allow for implicit labeling
                    indx_rl = [ind for ind in range(len(pos[0])) if pos[0][ind][0] in self.rl_ids][0]
                    indx_sorted_ids = np.mod(np.arange(len(pos[0])) + indx_rl, len(pos[0]))
                    sorted_ids += [pos[0][ind][0] for ind in indx_sorted_ids]
                    sorted_pos += [pos[0][ind][1] % self.scenario.length for ind in indx_sorted_ids]
                else:
                    sorted_ids += [tup[0] for tup in pos[i]]
                    sorted_pos += [tup[1] % self.scenario.length for tup in pos[i]]
            else:
                sorted_ids += [tup[0] for tup in pos[i]]
                sorted_pos += [tup[1] for tup in pos[i]]

        edge_id = [0] * len(pos[0]) + [1] * len(pos[1])
        if self.scenario.merge_out_len is not None:
            edge_id += [2] * len(pos[2])

        # the extra data in this case is a tuple of sorted positions and route ids
        sorted_extra_data = (sorted_pos, edge_id)

        return sorted_ids, sorted_extra_data

    def apply_acceleration(self, veh_ids, acc):
        """
        See parent class
        In addition, merging vehicles travel at the target velocity at the merging lanes,
        and vehicle that are about to leave the network stop.
        """
        i_called = []  # index of vehicles in the list that have already received traci calls
        for i, veh_id in enumerate(veh_ids):
            if "merge" in veh_id:
                this_edge = self.vehicles[veh_id]["edge"]
                this_pos = self.vehicles[veh_id]["position"]

                # vehicles that are about to exit are stopped
                if this_edge == "merge_out" and this_pos > self.scenario.merge_out_len - 10:
                    self.traci_connection.vehicle.slowDown(veh_id, 0, 1)
                    i_called.append(i)

                # vehicles in the merging lanes move at the target velocity (if one is defined)
                elif "target_velocity" in self.env_params.additional_params and this_edge in ["merge_in", "merge_out"]:
                    self.traci_connection.vehicle.slowDown(veh_id, self.env_params.get_additional_param("target_velocity"), 1)
                    i_called.append(i)

        # delete data on vehicles that have already received traci calls in order to reduce run-time
        veh_ids = [veh_ids[i] for i in range(len(veh_ids)) if i not in i_called]
        acc = [acc[i] for i in range(len(acc)) if i not in i_called]

        super().apply_acceleration(veh_ids=veh_ids, acc=acc)
