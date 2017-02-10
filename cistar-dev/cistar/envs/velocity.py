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

        thisSpeed = float(traci.vehicle.getSpeed(car_id))
        nextVel = thisSpeed + action * self.timestep
        traci.vehicle.slowDown(car_id, nextVel, 1)

    def render(self):
        pass


    def get_x_by_id(self, id):
        return self.scenario.get_x(self.vehicles[id]["edge"], self.vehicles[id]["position"])


    def get_cars(self, id, num_back=None, num_forward=None,
                 distance_back=None, distance_forward=None,
                 lane=None):
        ret = []
        target_pos = self.get_x_by_id(id)

        for i in self.ids:
            if i != id:
                c = self.vehicles[i]
                if (distance_back is not None and (target_pos - self.get_x_by_id(i)) % self.length > distance_back) or \
                        (num_back is not None and len(ret) >= num_back):
                    break
                if (lane is None or c["lane"] == lane):
                    ret.insert(0, c["id"])

            count = len(ret)

        for i in self.ids:
            if i != id:
                c = self.vehicles[i]
                if (distance_forward is not None and (self.get_x_by_id(i) - target_pos) % self.scenario.length > distance_forward) or \
                        (num_forward is not None and (len(ret) - count) >= num_forward):
                    break
                if (lane is None or c["lane"] == lane):
                    ret.append(c["id"])

        return ret

    def get_leading_car(self, id):
        return self.get_cars(id, num_forward=1, lane=self.vehicles["lane"])

    def get_trailing_car(self, id):
        return self.get_cars(id, num_back=1, lane=self.vehicles["lane"])

