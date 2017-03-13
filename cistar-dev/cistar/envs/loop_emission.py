from cistar.envs.loop import LoopEnvironment

from rllab.spaces import Box
from rllab.spaces import Product

import traci

import numpy as np



class SimpleEmissionEnvironment(LoopEnvironment):

    @property
    def action_space(self):
        """
        Actions are a set of velocities from -5 to 5m/s
        :return:
        """
        #TODO: max and min are parameters
        #TODO: Make more realistic
        return Box(low=-5, high=5, shape=(self.scenario.num_rl_vehicles, ))

    @property
    def observation_space(self):
        """
        See parent class
        TODO(Leah): Fill in documentation
        """
        # num_cars = self.tot_cars if self.fullbool else self.num_cars
        num_cars = self.scenario.num_vehicles
        ypos = Box(low=0., high=np.inf, shape=(num_cars, ))
        vel = Box(low=0., high=np.inf, shape=(num_cars, ))
        xpos = Box(low=0., high=np.inf, shape=(num_cars, ))
        fuelconsump = Box(low=0., high=np.inf, shape=(num_cars, ))
        return Product([vel, fuelconsump, ypos, xpos])

    def compute_reward(self, state):
        """
        See parent class
        TODO(Leah): Fill in documentation
        """
        destination = self.highway_length * 4
        return -np.sum(0.1*state[2] + 0.9*(destination - state[3]))

    def get_lane_position(self, vehicle):
        #TODO: Don't hardcode lanestarts in this function
        lanestarts = {"left": 3 * edgelen,
                       "top": 2*edgelen,
                       "right": edgelen,
                       "bottom": 0}

        vID = vehicle["id"]
        lanepos = vehicle["position"]
        lane = traci.vehicle.getLaneID(vID).split("_")
        return lanestarts[lane[0]] + lanepos

    def getState(self):
        """
        Acts as updateState
        TODO(Leah): Fill in documentation
        self.vehicles is a dictionary, each vehicle inside is a dictionary
        Cumulative distance = last cumulative dist + new pos - last pos
        """
        # new_speed = np.array([self.vehicles[vehicle]["speed"] for vehicle in self.vehicles])
        last_dist = np.copy(self._state[3])
        old_pos = -np.copy(self._state[1])

        self._state = np.array([[self.vehicles[vehicle]["speed"], \
                                self.get_lane_position(vehicle), \
                                self.vehicles[vehicle]["fuel"], \
                                0] for vehicle in self.vehicles]).T

        # Delta position change (distance travelled in last timestep)
        self._state[3] += self._state[1] + old_pos
        # If delta position is negative, that means you circled the loop
        self._state[3, self._state[3] < 0] += self.highway_length
        self._state[3] += last_dist
        return self._state

    def apply_action(self, car_id, action):
        """
        Action is an acceleration here. Gets locally linearized to find velocity.
        """
        traci.vehicle.slowDown(car_id, action, 1)

    def render(self):
        print('current velocity, fuel, distance:', self._state)