from cistar.core.base_env import SumoEnvironment

from rllab.spaces import Box

import traci

import numpy as np



class LoopEnvironment(SumoEnvironment):

    def get_last_x_by_id(self, id):
        return self.scenario.get_x(self.last_step[id]["edge"], self.last_step[id]["position"])

    def get_x_by_id(self, id):
        if self.vehicles[id]["edge"] == '':
            print("This vehicle teleported and its edge is now empty", id)
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

    def get_cars(self, car_id, dxBack, dxForward, lane = None, dx = None):
        this_pos = self.get_x_by_id(car_id) # position of the car checking neighbors
        front_limit = this_pos + dxForward
        rear_limit = this_pos - dxBack

        if dx == None:
            dx = .5 * (dxBack + dxForward)

        cars = []
        for i in self.ids:
            if i != car_id:
                car = self.vehicles[i]
                if lane is None or car['lane'] == lane:
                    # if a one-lane case or the correct lane
                    other_pos = self.get_x_by_id(i)
                    # if ((front_limit - other_pos) % self.scenario.length > 0) \
                    #     and ((other_pos - rear_limit) % self.scenario.length > 0):

                    # too lazy right now to differentiate between front/back distances
                    if (this_pos - other_pos) % self.scenario.length < dx:
                        cars.append(car['id'])

        return cars
