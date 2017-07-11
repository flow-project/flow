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
        return rewards.desired_velocity(
            state, rl_actions, fail=kwargs["fail"], target_velocity=self.env_params["target_velocity"])

    def getState(self, **kwargs):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        # if kwargs["observability"] == "full":
        # full observability
        return np.array([[self.vehicles[veh_id]["speed"] + normal(0, self.observation_vel_std),
                          self.vehicles[veh_id]["absolute_position"] + normal(0, self.observation_pos_std)]
                         for veh_id in self.sorted_ids]).T

        # else:
        #     # partial observability (n car ahead, m car behind)
        #     sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]
        #     veh_ids = []
        #     for veh_id in sorted_rl_ids:
        #         veh_ids.append(veh_id)  # add rl vehicle
        #         veh_ids.append(self.vehicles[veh_id]["leader"])  # add vehicle in front of rl vehicle
        #         veh_ids.append(self.vehicles[veh_id]["follower"])  # add vehicle behind rl vehicle
        #
        #     veh_ids = np.unique(veh_ids)  # remove redundant vehicle ids



        # partial observability (2 cars ahead, 2 cars behind)

    def render(self):
        print('current state/velocity:', self.state)
