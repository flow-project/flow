from cistar.core.base_env import SumoEnvironment

from rllab.spaces import Box

import traci

import numpy as np
import random
import logging


class SimpleVelocityEnvironment(SumoEnvironment):

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

        thisSpeed = float(traci.vehicle.getSpeed(car_id))
        nextVel = thisSpeed + action * self.timestep
        traci.vehicle.slowDown(car_id, nextVel, 1)

    def render(self):
        pass
