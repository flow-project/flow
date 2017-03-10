from cistar.envs.loop import LoopEnvironment

from rllab.spaces import Box

import traci

import numpy as np


"""
Fully functional environment. Takes in an *acceleration* as an action. Reward function is negative norm of the
difference between the velocities of each vehicle, and the target velocity. State function is a vector of the
velocities for each vehicle.
"""
class SimpleAccelerationEnvironment(LoopEnvironment):


    @property
    def action_space(self):
        """
        Actions are a set of accelerations from 0 to 15m/s
        :return:
        """
        #TODO: max and min are parameters
        return Box(low=self.env_params["max-deacc"], high=self.env_params["max-acc"], shape=(self.scenario.num_rl_vehicles, ))

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        return Box(low=-np.inf, high=np.inf, shape=(self.scenario.num_vehicles, ))

    def apply_action(self, car_id, action):
        """
        See parent class (base_env)
         Given an acceleration, set instantaneous velocity given that acceleration.
        """
        thisSpeed = self.vehicles[car_id]['speed']
        nextVel = thisSpeed + action * self.time_step
        nextVel = max(0, nextVel)
        traci.vehicle.slowDown(car_id, nextVel, 1)

    def compute_reward(self, velocity):
        """
        See parent class
        """
        return -np.linalg.norm(velocity - self.env_params["target_velocity"])

    def getState(self):
        """
       See parent class
       The state is an array the velocities for each vehicle
       :return: an array of vehicle speed for each vehicle
       """
        return np.array([self.vehicles[vehicle]["speed"] for vehicle in self.vehicles])

    def render(self):
        print('current state/velocity:', self._state)