from cistar.envs.loop import LoopEnvironment

from rllab.spaces import Box
from rllab.spaces import Product

import traci

import numpy as np
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
        #TODO: max and min are parameters
        return Box(low=-np.abs(self.env_params["max-deacc"]), high=self.env_params["max-acc"],
                   shape=(self.scenario.num_rl_vehicles, ))

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        speed = Box(low=-np.inf, high=np.inf, shape=(self.scenario.num_vehicles,))
        headway = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        pos = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        return Product([speed, pos])

    def apply_rl_actions(self, rl_actions):
        """
        See parent class
        """
        sorted_indx = np.argsort([self.vehicles[veh_id]["absolute_position"] for veh_id in self.rl_ids])
        sorted_rl_ids = np.array(self.rl_ids)[sorted_indx]

        self.apply_acceleration(sorted_rl_ids, rl_actions)
        # self.apply_acceleration(self.rl_ids, rl_actions)

        # target_lane = None
        # lane_change_penalty = None
        #
        # return actual_acc, acc_deviation, target_lane, lane_change_penalty

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        if any(state[0] < 0) or kwargs["fail"]:
            return -20.0

        reward_type = 'speed'

        if reward_type == 'speed':
            max_cost = np.array([self.env_params["target_velocity"]]*self.scenario.num_vehicles)
            max_cost = np.linalg.norm(max_cost)

            cost = state[0] - self.env_params["target_velocity"]
            cost = np.linalg.norm(cost)

            return max(max_cost - cost, 0)

        elif reward_type == 'distance':
            distance = np.array([self.vehicles[veh_id]["absolute_position"] - self.initial_pos[veh_id]
                                 for veh_id in self.ids])

            return sum(distance)

    def getState(self):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        sorted_indx = np.argsort([self.vehicles[veh_id]["absolute_position"] for veh_id in self.ids])
        sorted_ids = np.array(self.ids)[sorted_indx]

        return np.array([[self.vehicles[vehicle]["speed"],
                          self.vehicles[vehicle]["absolute_position"]] for vehicle in sorted_ids]).T

        # return np.array([[self.vehicles[vehicle]["speed"],
        #                   self.get_headway(vehicle)] for vehicle in sorted_ids]).T

        # return np.array([[self.vehicles[vehicle]["speed"],
        #                   self.vehicles[vehicle]["absolute_position"]] for vehicle in self.vehicles]).T

        # return np.array([[self.vehicles[vehicle]["speed"],
        #                   self.get_headway(vehicle)] for vehicle in self.ids]).T

    def render(self):
        print('current state/velocity:', self.state)
