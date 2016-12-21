import numpy as np
# import matplotlib as plt
from rllab.envs.base import Env
from rllab.spaces import Box
# from rllab.spaces import Product
from rllab.envs.base import Step

import sumo_config as defaults
from sumolib import checkBinary

import subprocess, sys
import traci
import traci.constants as tc


"""
This file contains the ExperimentEnv class, which extends SumoEnvironment and
is to be created every time you specfiy a new type of experiment to use cistar with.

The methods on SumoEnvironment are general to all experiments, and handle the interactions
with the SUMO/TraCI simulation. The methods defined here are specific to your particular
experiment, and deal with the MDP and other parameters you may wish to change frequently.

Note that you may edit any of the functions that end in "pass"

"""

class ExperimentEnv(SumoEnvironment):

	def __init__(self, num_vehicles, env_params, vehicle_controllers, sumo_binary, sumo_params):
		SumoEnvironment.__init__(self)
        pass


    def apply_action(self, car_id, action):
        SumoEnvironment.apply_action(self, car_id, action)
        raise NotImplementedError

	def step(self, car_actions):
		"""
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        SumoEnvironment.step(self, car_actions)

        for car_id, controller, in vehicle_controllers.items():
            action = car_actions[car_id]
            traci.vehicle.slowDown(self.controllable[car_id], apply_action(car_id, action), 1)



		# for car in self.robots:
		# 	pass # Call traci function to move the car according to action input 
		# for car in self.humans:
		# 	pass #  Use controller to determine next action
		traci.simulationStep()

		self._state = np.array([traci.vehicle.getSpeed(vID) for vID in self.controllable])
		reward = self.compute_reward(self._state)
		next_observation = np.copy(self._state)
		return Step(observation=next_observation, reward=reward, done=False)


	def reset(self):
		"""
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        SumoEnvironment.reset(self)
        # traci method to move all cars back to initial state


        # self state is velocity, observation is velocity
        # self._state = np.random.uniform(0, self.GOAL_VELOCITY, size=(self.num_cars,))
        # print("In reset function", self.initialized)



    # @property
    # def action_space(self):
    #     """
    #     Returns a Space object
    #     """
    #     return Box(low=-5, high=5, shape=(self.num_cars, ))

    # @property
    # def observation_space(self):
    #     """
    #     Returns a Space object
    #     """
    #     return Box(low=-np.inf, high=np.inf, shape=(self.num_cars,))
    #     # pos = Box(low=0., high=self.road_length, shape=(self.num_cars, ))
    #     # vel = Box(low=0., high=self.GOAL_VELOCITY+10, shape=(self.num_cars, ))
    #     # accel = Box(low=self.min_acceleration, high=self.max_acceleration, shape=(self.num_cars, ))
    #     # return vel
    #     # return Product([pos, vel, accel])


    def set_command(self, updated):
        SumoEnvironment.set_command(self, updated)
		pass # Convert actions from rllab into commands for robot cars
		# Set functions for each car appropriately

	def compute_reward(self, state, action):
        SumoEnvironment.compute_reward(self, state, action)
		return 0

    def render(self):
        SumoEnvironment.render(self)
        print('current state/velocity:', self._state)

    def terminate(self):
        SumoEnvironment.terminate(self)
        traci.close()



