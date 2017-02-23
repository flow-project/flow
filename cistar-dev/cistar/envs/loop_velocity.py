from cistar.core.base_env import SumoEnvironment

from rllab.spaces import Box

import traci

import numpy as np



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
        traci.vehicle.slowDown(car_id, action, 1)

    def render(self):
        pass

    def get_last_x_by_id(self, id):
        return self.scenario.get_x(self.last_step[id]["edge"], self.last_step[id]["position"])

    def get_x_by_id(self, id):
        return self.scenario.get_x(self.vehicles[id]["edge"], self.vehicles[id]["position"])

    def get_leading_car(self, car_id, lane = None):
        target_pos = self.get_x_by_id(car_id)

        frontdists = []
        for i in self.ids:
            if i != car_id:
                c = self.vehicles[i]
                if lane is None or c['lane'] == lane:
                    distto = (self.get_x_by_id(i) - target_pos) % self.scenario.length
                    frontdists.append((c["id"], distto))

        if frontdists:
            return min(frontdists, key=(lambda x:x[1]))[0]
        else:
            return None

    def get_trailing_car(self, car_id, lane = None):
        target_pos = self.get_x_by_id(car_id)

        backdists = []
        for i in self.ids:
            if i != car_id:
                c = self.vehicles[i]
                if lane is None or c['lane'] == lane:
                    distto = (target_pos - self.get_x_by_id(i)) % self.scenario.length
                    backdists.append((c["id"], distto))

        if backdists:
            return min(backdists, key=(lambda x:x[1]))[0]
        else:
            return None
