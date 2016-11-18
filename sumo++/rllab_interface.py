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
This file contains methods for sumo++ interactions with rllab
Most notably, it has the SumoEnvironment class, which serves as the environment
that gets passed in to rllab's algorithm (it's basically an MDP)

When defining a SumoEnvironment, one must still implement the action space, 
observation space, reward, etc. (properties of the MDP). This class is meant to 
serve as a parent.

"""

class SumoEnvironment(Env):

	delta = 0.01
	PORT = defaults.PORT
    sumoBinary = checkBinary(defaults.BINARY)

    self.num_cars
	self.env_params
	self.initial_state

	self.robots # Indices of robot cars
	self.car_functions
	self.human_policy


	def __init__(self, num_vehicles, env_params, vehicle_controllers, sumo_binary, sumo_params):
		Env.__init__(self)
        self.env_params = env_params
        self.human_ids = [i for i in range(num_vehicles) if i in vehicle_controllers.keys()]

		# Set all params above as necessary
		traci.init(self.PORT)
		sumoProcess = subprocess.Popen([self.sumoBinary, "-c", self.cfgfn, "--remote-port", \
										str(self.PORT)], stdout=sys.stdout, stderr=sys.stderr)

		self.vehIDs = traci.vehicle.getIDList()
        self._state = np.array([traci.vehicle.getSpeed(vID) for vID in self.controllable])
        observation = np.copy(self._state)
        return observation        


    def apply_action(self, car_id, action):
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
		pass # Convert actions from rllab into commands for robot cars
		# Set functions for each car appropriately

	def compute_reward(self, state, action):
		return 0

    def render(self):
        print('current state/velocity:', self._state)

    def terminate(self):
        traci.close()



