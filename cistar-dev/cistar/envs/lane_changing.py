from cistar.envs.loop import LoopEnvironment

from rllab.spaces import Box
from rllab.spaces import Product
import traci

import numpy as np


"""
Fully functional environment. Takes in an *acceleration* as an action. Reward function is negative norm of the
difference between the velocities of each vehicle, and the target velocity. State function is a vector of the
velocities for each vehicle.
"""
class SimpleLaneChangingAccelerationEnvironment(LoopEnvironment):


    @property
    def action_space(self):
        """
        Actions are a set of accelerations from 0 to 15m/s
        :return:
        """
        #TODO: max and min are parameters
        # accelerations = Box(low=, high=self.env_params["max-acc"], shape=(self.scenario.num_rl_vehicles, ))
        # lc_threshold = Box(low=-1, high=1, shape=(self.scenario.num_rl_vehicles, ))
        # left_right_threshold = Box(low=-1, high=1, shape=(self.scenario.num_rl_vehicles, ))

        lb = [self.env_params["max-deacc"], -1, -1] * self.scenario.num_rl_vehicles
        ub = [self.env_params["max-deacc"], 1, 1] * self.scenario.num_rl_vehicles
        return Box(np.array(lb), np.array(ub))

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
        # if we're being completely mathematically correct, 1 should be replaced by int(self.time_step * 1000)
        # but it shouldn't matter too much, because 1 is always going to be less than int(self.time_step * 1000)
        traci.vehicle.slowDown(car_id, nextVel, 1)

    def compute_reward(self, velocity, action):
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
        print('current state/velocity:', self.state)

    def apply_rl_actions(self, actions):
        for i ,veh_id in enumerate(self.rl_ids):
            lc_threshold = actions[3 * i + 1]
            direction = actions[3 * i + 2]
            if lc_threshold > 0:
                if direction > 0:
                    # lane change right
                    self.vehicles[veh_id][]
                else:
                    # lane change left

            acceleration = actions[3*i]
            if self.fail_safe == 'instantaneous':
                safe_action = self.vehicles[veh_id]['controller'].get_safe_action_instantaneous(self, acceleration)
            elif self.fail_safe == 'eugene':
                safe_action = self.vehicles[veh_id]['controller'].get_safe_action(self, acceleration)
            else:
                safe_action = acceleration
            self.apply_action(veh_id, action=safe_action)