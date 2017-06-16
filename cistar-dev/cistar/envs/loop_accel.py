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
        absolute_pos = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        return Product([speed, absolute_pos])

    def apply_rl_actions(self, rl_actions):
        """
        See parent class
        """
        self.apply_acceleration(self.rl_ids, rl_actions)

        return []

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        if any(state[0] < 0) or kwargs["fail"]:
            return -20.0

        max_cost = np.array([self.env_params["target_velocity"]]*self.scenario.num_vehicles)
        max_cost = np.linalg.norm(max_cost)

        cost = state[0] - self.env_params["target_velocity"]
        cost = np.linalg.norm(cost)

        return max_cost - cost

    def getState(self):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        return np.array([[self.vehicles[vehicle]["speed"],
                          self.vehicles[vehicle]["absolute_position"]] for vehicle in self.vehicles]).T

    def render(self):
        print('current state/velocity:', self.state)
