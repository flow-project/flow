from cistar.envs.loop import LoopEnvironment
from cistar.core import rewards

from rllab.spaces import Box
from rllab.spaces import Product

import numpy as np
from numpy.random import normal

import pdb


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
        return Box(low=-np.abs(self.env_params["max-deacc"]), high=self.env_params["max-acc"],
                   shape=(self.scenario.num_rl_vehicles, ))

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        speed = Box(low=0, high=np.inf, shape=(self.scenario.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        return Product([speed, absolute_pos])

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
        vel_non_merge = np.array([self.vehicles[veh_id]["speed"] for veh_id in self.ids if "merge" not in veh_id])
        vel_merge = np.array([self.vehicles[veh_id]["speed"] for veh_id in self.ids if "merge" in veh_id])

        # check for crashes
        if any(np.append(vel_merge, vel_non_merge) < -100) or kwargs["fail"]:
            return 0

        # reward the velocity of vehicle in the ring w.r.t. target velocity
        max_cost = np.linalg.norm(np.array([self.env_params["target_velocity"]] * len(vel_non_merge)))
        cost = np.linalg.norm(vel_non_merge - self.env_params["target_velocity"])
        reward_non_merge = max(0, max_cost - cost)

        # reward the velocity of merging vehicles w.r.t. speed limit
        max_cost = np.linalg.norm(np.array([self.scenario.net_params["speed_limit"]] * len(vel_non_merge)))
        cost = np.linalg.norm(vel_non_merge - self.scenario.net_params["speed_limit"])
        reward_merge = max(0, max_cost - cost)

        return reward_non_merge + reward_merge

    def getState(self, **kwargs):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        pos = []
        for veh_id in self.sorted_ids:
            # the position of non-merging vehicles is their relative position on the ring
            if "merge" not in veh_id:
                pos.append(self.vehicles[veh_id]["absolute_position"] % self.scenario.length)
            # the position of merging vehicles is their absolute position in the network
            # (in the ring, it would be their relative position on the ring)
            else:
                pos.append(self.vehicles[veh_id]["absolute_position"])

        return np.array([[self.vehicles[veh_id]["speed"] + normal(0, self.observation_vel_std),
                          pos[i] + normal(0, self.observation_pos_std)]
                         for i, veh_id in enumerate(self.sorted_ids)]).T

        # return np.array([[self.vehicles[veh_id]["speed"] + normal(0, self.observation_vel_std),
        #                   self.vehicles[veh_id]["absolute_position"] + normal(0, self.observation_pos_std)]
        #                  for veh_id in self.sorted_ids]).T

    def render(self):
        print('current state/velocity:', self.state)

    def additional_command(self):
        """
        In the case when a vehicle meant to stay in the ring is about to reach a rout choosing node,
        this function reroutes the vehicle to keep in in the network
        """
        for veh_id in self.ids:
            # if the vehicle is one the merging vehicles, and there is a merge-out lane, it should not be rerouted
            if "merge" in self.vehicles[veh_id]["type"] and self.scenario.net_params["merge_out_length"] is not None:
                continue

            # TODO: add currents routes to the vehicles dict so that we don't have to reroute multiple times
            # check if a vehicle needs to be rerouted
            route = None
            if self.vehicles[veh_id]["edge"] == "ring_0":
                route = ["ring_0", "ring_1"]
            elif self.vehicles[veh_id]["edge"] == "ring_1":
                route = ["ring_1", "ring_0"]

            # perform rerouting
            if route is not None:
                self.traci_connection.vehicle.setRoute(vehID=veh_id, edgeList=route)

    def sort_by_position(self, **kwargs):
        """
        See parent class
        Vehicles in the ring are sorted by their relative position in the ring, while vehicles outside the ring
        are sorted according to their position of their respective edge.
        Vehicles are sorted by position on the ring, the in-merge, then the out-merge
        """
        # abs_pos = []
        # for veh_id in self.ids:
        #     # for not non-merging-vehicles, the absolute position is not changed
        #     if "merge" not in self.vehicles[veh_id]["type"]:
        #         abs_pos.append((veh_id, self.vehicles[veh_id]["absolute_position"]))
        #
        #     # for merging vehicles in the ring, the absolute position is augmented by the number of loops
        #     # the vehicle
        #     elif self.vehicles[veh_id]["edge"] not in ["merge_in", "merge_out"]:
        #         if 0 < self.vehicles[veh_id]["absolute_position"] < self.scenario.length:
        #             # closest vehicle behind the merge in node
        #             non_merge_vehicle_pos = [(veh_id, self.vehicles[veh_id]["absolute_position"])
        #                                      for veh_id in self.ids if "merge" not in veh_id]
        #             lag_id = max(non_merge_vehicle_pos, key=lambda x: x[1] % self.scenario.length)[0]
        #
        #             # number of loops performed by the lagging vehicle
        #             num_loops = int(self.vehicles[lag_id]["absolute_position"] / self.scenario.length)
        #
        #             self.vehicles[veh_id]["absolute_position"] += (num_loops + 1) * self.scenario.length
        #
        #         abs_pos.append((veh_id, self.vehicles[veh_id]["absolute_position"]))
        #
        #     # merging vehicles outside the ring have absolute positions equal to some constant plus their relative
        #     # position on the given edge
        #     else:
        #         if self.vehicles[veh_id]["edge"] == "merge_in":
        #             abs_pos.append((veh_id, 1000 * self.scenario.length + self.vehicles[veh_id]["position"]))
        #         else:
        #             abs_pos.append((veh_id, 1000 * self.scenario.length + self.vehicles[veh_id]["position"] +
        #                             self.scenario.merge_in_len))
        #
        # # sort absolute positions and collect ids
        # abs_pos.sort(key=lambda tup: tup[1])
        # sorted_ids = [tup[0] for tup in abs_pos]
        #
        # return sorted_ids

        # position of merging vehicles and non-merging vehicles are collected and ordered separately
        merge_veh = []
        non_merge_veh = []
        for veh_id in self.ids:
            if "merge" in veh_id:
                merge_veh.append((veh_id, self.vehicles[veh_id]["absolute_position"]))
            else:
                non_merge_veh.append((veh_id, self.vehicles[veh_id]["absolute_position"]))

        merge_veh.sort(key=lambda tup: tup[1])
        non_merge_veh.sort(key=lambda tup: tup[1])

        sorted_merge_ids = [tup[0] for tup in merge_veh]
        sorted_non_merge_ids = [tup[0] for tup in non_merge_veh]

        # the sorted ids consists of the sorted ids of the non-merging vehicles, followed by the sorted ids of the
        # merging vehicles
        sorted_ids = sorted_non_merge_ids + sorted_merge_ids

        return sorted_ids

    def apply_acceleration(self, veh_ids, acc):
        """
        See parent class
        In addition, merging vehicles that are about to leave the network stop
        """
        super().apply_acceleration(veh_ids=veh_ids, acc=acc)

        for veh_id in veh_ids:
            if "merge" in veh_id:
                this_edge = self.vehicles[veh_id]["edge"]
                this_pos = self.vehicles[veh_id]["position"]

                if this_edge == "merge_out" and this_pos > self.scenario.merge_out_len - 10:
                    self.traci_connection.vehicle.slowDown(vehID=veh_id, speed=0, duration=1)
