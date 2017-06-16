from cistar.envs.loop import LoopEnvironment

from rllab.spaces import Box
from rllab.spaces import Product

import traci

import numpy as np
import pdb



class SimpleEmissionEnvironment(LoopEnvironment):
    '''Environment subclass
    
    This class has a state space that is on velocity, fuel, and headway
    between cars. 
    
    Extends:
        LoopEnvironment
    '''

    @property
    def action_space(self):
        """
        Actions are a set of velocities from -5 to 5m/s
        :return:
        """
        #TODO: max and min are parameters
        #TODO: Make more realistic
        return Box(low=-abs(self.env_params["max-deacc"]), high=3.0, shape=(self.scenario.num_rl_vehicles, ))

    @property
    def observation_space(self):
        """
        See parent class
        TODO(Leah): Fill in documentation
        """
        # num_cars = self.tot_cars if self.fullbool else self.num_cars
        num_cars = self.scenario.num_vehicles
        #ypos = Box(low=0., high=np.inf, shape=(num_cars, ))
        vel = Box(low=0., high=np.inf, shape=(num_cars, ))
        abs_pos = Box(low=0., high=np.inf, shape=(num_cars, ))
        #fuelconsump = Box(low=0., high=np.inf, shape=(num_cars, ))
        self.obs_var_labels = ["Velocity", "Absolute Position"]
        return Product([vel, abs_pos])

    def compute_reward(self, state, action, **kwargs):
        """
        See parent class
        TODO(Leah): Fill in documentations
        """
        destination = 840 * 4
        #return -np.sum(0.1*state[1] + 3.0*(destination - state[2]))
        #return -np.sum(destination - state[3])
        #return -np.linalg.norm(state[0] - self.env_params["target_velocity"])
        #return np.mean(state[0] - self.env_params["target_velocity"])

        if any(state[0] < 0):
            print('crashed and neg value')
            return -20.0
        max_cost = np.array([self.env_params["target_velocity"]]*self.scenario.num_vehicles)
        max_cost = np.linalg.norm(max_cost)

        cost = state[0] - self.env_params["target_velocity"]
        cost = np.linalg.norm(cost)
        return max_cost - cost

    def getState(self):

        #return np.array([[self.vehicles[veh_id]["speed"], \
        #                    self.vehicles[veh_id]["fuel"], \
        #                    self.get_headway(veh_id)] for veh_id in self.vehicles]).T

        return np.array([[self.vehicles[veh_id]["speed"], \
                           self.vehicles[veh_id]["absolute_position"]] for veh_id in self.vehicles]).T


    def apply_action(self, car_id, action):
        """
        See parent class (base_env)
         Given an acceleration, set instantaneous velocity given that acceleration.
        """
        thisSpeed = self.vehicles[car_id]['speed']
        nextVel = thisSpeed + action * self.time_step
        nextVel = max(0, nextVel)
        # if we're being completely mathematically correct, 1 should be replaced by int(self.time_step * 1000)
        # but it shouldn't matter too much, because 1 is always going to be less than int(self.time_step * 1000)
        self.traci_connection.vehicle.slowDown(car_id, nextVel, 1)

    def render(self):
        print('current velocity, headway', self.state)