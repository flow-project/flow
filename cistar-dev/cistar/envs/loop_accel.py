from cistar.envs.loop import LoopEnvironment

from rllab.spaces import Box

import traci

import numpy as np



class SimpleAccelerationEnvironment(LoopEnvironment):

    @property
    def action_space(self):
        return Box(low=-5, high=5, shape=(len(self.rl_ids), ))

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(len(self.ids), ))

    def compute_reward(self, velocity):
        return -np.linalg.norm(velocity - self.env_params["target_velocity"])

    def apply_action(self, car_id, action):
        '''Action is an acceleration here. Gets locally linearized to find velocity.'''
        thisSpeed = self.vehicles[car_id]['speed']
        nextVel = thisSpeed + action * self.time_step
        nextVel = max(0, nextVel)
        traci.vehicle.slowDown(car_id, nextVel, 1)

    def render(self):
        print('current state/velocity:', self._state)