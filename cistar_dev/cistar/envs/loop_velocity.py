from cistar.envs.loop import LoopEnvironment

from gym.spaces.box import Box

import traci

import numpy as np


class SimpleVelocityEnvironment(LoopEnvironment):

    @property
    def action_space(self):
        """
        Actions are a set of velocities from 0 to 15m/s
        :return:
        """
        return Box(low=self.env_params.get_additional_param("min-vel"),
                   high=self.env_params.get_additional_param("max-vel"),
                   shape=(self.scenario.num_rl_vehicles,))

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        self.obs_var_labels = ["Velocity"]
        return Box(low=-np.inf, high=np.inf, shape=(self.vehicles.num_vehicles,))

    def compute_reward(self, state, actions, **kwargs):
        """
        See parent class
        """
        velocity = state
        return -np.linalg.norm(velocity - self.env_params.get_additional_param("target_velocity"))

    def get_state(self):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: an array of vehicle speed for each vehicle
        """
        return np.array(self.vehicles.get_speed(self.ids))

    def apply_action(self, car_id, action):
        """
        Action is an velocity here.
        """
        not_zero = max(0, action)
        self.traci_connection.vehicle.slowDown(car_id, not_zero, 1)

    # def render(self):
    #     pass
