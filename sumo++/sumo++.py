from controller_interface.py import ____
from rllab_interface.py import ____


"""
Primary sumo++ file, imports API from supporting files and manages interactions 
with rllab and custom controllers.

In addition to opening a traci port and running an instance of Sumo, the 
simulation class should store the controllers for both the manual and the 
autonomous vehicles, which it will use implement actions on the vehicles.

Interfaces with sumo on the other side

"""


class SumoSimulation():

	def __init__(self, params):
		self.env_params
		self.initial_state

		self.robots # Indices of robot cars
		self.car_functions

		traci.init(self.PORT)

		self.human_policy



		pass


	def set_policy(self, updated):
		pass # Convert actions from rllab into commands for robot cars
		# Set functions for each car appropriately


	def step(self, car_actions):
		for car in self.robots:
			pass # Call traci functions to move the cars according to policy input 

		for car in self.humans:
			pass 


	def get_state(self):
		pass


	def reset(self):
		pass # resets to initial state
