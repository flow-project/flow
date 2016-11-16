from controller_interface.py import HumanController, RobotController
from rllab_interface.py import SumoEnvironment


"""
Primary sumo++ file, imports API from supporting files and manages interactions 
with rllab and custom controllers.

In addition to opening a traci port and running an instance of Sumo, the 
simulation class should store the controllers for both the manual and the 
autonomous vehicles, which it will use implement actions on the vehicles.

Interfaces with sumo on the other side

"""


class SumoExperiment():

	self.env_params
	self.initial_state

	self.robots # Indices of robot cars
	self.car_functions
	self.human_policy


	def __init__(self, params):
		
		env = SumoEnvironment(env_params)

