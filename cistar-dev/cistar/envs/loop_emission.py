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
        return Box(low=0, high=5, shape=(self.scenario.num_rl_vehicles, ))

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
        self.obs_var_labels = ["Velocity", "Fuel", "Distance"]
        return Product([vel, fuelconsump, ypos])

    def compute_reward(self, state, action):
        """
        See parent class
        TODO(Leah): Fill in documentation
        """
        destination = 840 * 4
        return -np.sum(0.1*state[1] + 0.9*(destination - state[2]))

    def getState(self):
        return np.array([[self.vehicles[veh_id]["speed"], \
                            self.vehicles[veh_id]["fuel"], \
                            self.vehicles[veh_id]["distance"]] for veh_id in self.vehicles]).T

    def apply_action(self, car_id, action):
        """
        Action is an acceleration here. Gets locally linearized to find velocity.
        """
        not_zero = max(0, action)
        self.traci_connection.vehicle.slowDown(car_id, not_zero, 1)

    def render(self):
        print('current velocity, fuel, distance:', self.state)