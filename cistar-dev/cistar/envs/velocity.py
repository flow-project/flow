from cistar.core.base_env import SumoEnvironment

from rllab.spaces import Box

import traci

import numpy as np



class SimpleVelocityEnvironment(SumoEnvironment):

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


    def get_x_by_id(self, id):
        if self.vehicles[id]["edge"] == '':
            print("This vehicle teleported and its edge is now empty", id)
        return self.scenario.get_x(self.vehicles[id]["edge"], self.vehicles[id]["position"])

    def get_leading_car(self, id):
        target_pos = self.get_x_by_id(id)

        frontdists = []
        for i in self.ids:
            if i != id:
                c = self.vehicles[i]
                distto = (self.get_x_by_id(i) - target_pos) % self.scenario.length
                frontdists.append((c["id"], distto))

        return min(frontdists, key=(lambda x:x[1]))[0]

    def get_trailing_car(self, id):
        target_pos = self.get_x_by_id(id)

        backdists = []
        for i in self.ids:
            if i != id:
                c = self.vehicles[i]
                distto = (target_pos - self.get_x_by_id(i)) % self.scenario.length
                backdists.append((c["id"], distto))

        return min(backdists, key=(lambda x:x[1]))[0]

