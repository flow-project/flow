from cistar.envs.loop import LoopEnvironment

from rllab.spaces import Box

import traci

import numpy as np



class SimpleVelocityEnvironment(LoopEnvironment):

    @property
    def action_space(self):
        return Box(low=-5, high=5, shape=(2,))

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    def compute_reward(self, velocity):
        return -np.linalg.norm(velocity - self.env_params["target_velocity"])

    def apply_action(self, car_id, action):
        '''Action is an acceleration here. Gets locally linearized to find velocity.'''
        traci.vehicle.slowDown(car_id, action, 1)

    def render(self):
        pass