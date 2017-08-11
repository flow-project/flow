from cistar.envs.loop import LoopEnvironment
from cistar.core import rewards

from rllab.spaces import Box
from rllab.spaces import Product

import traci

import numpy as np
from numpy.random import normal

import pdb


class SimpleAccelerationEnvironment(LoopEnvironment):
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

        # # partial observability
        # speed = Box(low=0, high=np.inf, shape=(3,))
        # absolute_pos = Box(low=0., high=np.inf, shape=(3,))
        # return Product([speed, absolute_pos])

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
        # reward = rewards.desired_velocity(
        #     state, rl_actions, fail=kwargs["fail"], target_velocity=self.env_params["target_velocity"])

        # reward desired velocity
        vel = np.array([self.vehicles[veh_id]["speed"] for veh_id in self.ids])
        if any(vel < -100) or kwargs["fail"]:
            return 0.
        max_cost = np.array([self.env_params["target_velocity"]] * self.scenario.num_vehicles)
        max_cost = np.linalg.norm(max_cost)
        cost = vel - self.env_params["target_velocity"]
        cost = np.linalg.norm(cost)
        reward = max(max_cost - cost, 0)

        # punish small headways
        headway_threshold = 30
        penalty_gain = 0.4
        penalty_exponent = 2
        headway_penalty = 0
        for veh_id in self.rl_ids:
            if self.vehicles[veh_id]["headway"] < headway_threshold:
                headway_penalty += (((headway_threshold - self.vehicles[veh_id]["headway"]) / headway_threshold)
                                    ** penalty_exponent) * penalty_gain

        # in order to keep headway penalty (and thus reward function) positive
        max_headway_penalty = self.scenario.num_rl_vehicles * penalty_gain
        headway_penalty = max_headway_penalty - headway_penalty

        reward += headway_penalty

        return reward

    def getState(self, **kwargs):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        # return np.array([[self.vehicles[veh_id]["speed"] + normal(0, self.observation_vel_std),
        #                   self.vehicles[veh_id]["absolute_position"] + normal(0, self.observation_pos_std)]
        #                  for veh_id in self.sorted_ids]).T

        """implicit labeling for stabilizing the ring (centering the rl vehicle, scaling and using relative position)"""
        scaled_rel_pos = [(self.vehicles[veh_id]["absolute_position"] % self.scenario.length) / self.scenario.length
                          for veh_id in self.sorted_ids]
        scaled_vel = [self.vehicles[veh_id]["speed"] / self.env_params["target_velocity"]
                      for veh_id in self.sorted_ids]
        return np.array([[scaled_vel[i], scaled_rel_pos[i]] for i in range(len(self.sorted_ids))]).T

        # """for purely homogeneous cases (i.e. full autonomy)"""
        # # note: for mixed-autonomy with more than one rl car, the reference vehicle can probably be chosen
        # # differently during the run, and still support the concept of equivalent classes
        # scaled_rel_pos = [self.vehicles[veh_id]["absolute_position"] / self.scenario.length
        #                   for veh_id in self.sorted_ids]
        # scaled_vel = [self.vehicles[veh_id]["speed"] / self.env_params["target_velocity"]
        #               for veh_id in self.sorted_ids]
        # return np.array([[scaled_vel[i], scaled_rel_pos[i]] for i in range(len(self.sorted_ids))]).T

    def render(self):
        print('current state/velocity:', self.state)


class SimplePartiallyObservableEnvironment(SimpleAccelerationEnvironment):
    """
    This environment is an extension of the SimpleAccelerationEnvironment (seen above), with the exception
    that only partial information is provided to the agent about the network; namely, information on the
    vehicle immediately in front of it and the vehicle immediately behind it. The reward function, however,
    continues to reward GLOBAL performance. The environment also assumes that only one autonomous vehicle
    is in the network.
    """
    # TODO: maybe generalize for several vehicles?

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        # partial observability
        speed = Box(low=0, high=np.inf, shape=(3,))
        absolute_pos = Box(low=0., high=np.inf, shape=(3,))
        return Product([speed, absolute_pos])

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        # reward desired velocity
        vel = np.array([self.vehicles[veh_id]["speed"] for veh_id in self.ids])
        if any(vel < -100) or kwargs["fail"]:
            return 0.

        max_cost = np.array([self.env_params["target_velocity"]] * self.scenario.num_vehicles)
        max_cost = np.linalg.norm(max_cost)

        cost = vel - self.env_params["target_velocity"]
        cost = np.linalg.norm(cost)

        return max(max_cost - cost, 0)

    def getState(self, **kwargs):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        vehID = self.rl_ids[0]
        lead_id = self.vehicles[vehID]["leader"]
        trail_id = self.vehicles[vehID]["follower"]

        if trail_id is None:
            trail_id = vehID
            self.vehicles[trail_id]["headway"] = 0
        if lead_id is None:
            lead_id = vehID
            self.vehicles[vehID]["headway"] = 0

        # state contains the speed of the rl car, and its leader and follower,
        # as well as the rl car's position in the network, and its headway with the vehicles adjacent to it
        observation = np.array([
            [self.vehicles[vehID]["speed"],
             self.vehicles[trail_id]["speed"],
             self.vehicles[lead_id]["speed"]],
            [self.vehicles[vehID]["absolute_position"] / self.scenario.length,
             self.vehicles[trail_id]["headway"] / self.scenario.length,
             self.vehicles[vehID]["headway"] / self.scenario.length]])

        return observation
