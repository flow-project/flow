import logging

import numpy as np
# import matplotlib as plt
from rllab.envs.base import Env
from rllab.spaces import Box
# from rllab.spaces import Product
from rllab.envs.base import Step

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

    def __init__(self, num_vehicles, num_rl_vehicles, env_params,
                 vehicle_controllers, sumo_binary, sumo_params, initial_config):
        """
        Initialize the Sumo Environment, by starting SUMO, setting up TraCI and initializing vehicles
        Input
        -----
        num_vehicles : total number of vehicles, RL and controller based
        env_params   : use this dictionary to pass in parameters relevant to the environment
                     (i.e. shape, size of simulatiomn; constants for step functions, etc. )
        vehicle_controllers : dictionary of car -> controller assignments for non-RL cars
        sumo_binary : SUMO library to start
        sumo_params : port, config file, out, error etc.


        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """

        self.num_vehicles = num_vehicles
        self.num_rl_vehicles = num_rl_vehicles
        self.env_params = env_params
        self.vehicle_controllers = vehicle_controllers
        self.sumo_binary = sumo_binary
        self.initial_config = initial_config

        if "port" not in sumo_params:
            raise logging.error("SUMO port not defined")
        else:
            self.port = sumo_params["port"]

        self.cfg = sumo_params['cfg']


        # could possibly be handled in sumo experiment

        self.rl_ids = [i for i in self.vehicle_controllers if vehicle_controllers[i] is None]
        self.controlled_ids = [i for i in self.vehicle_controllers if vehicle_controllers[i] is not None]
        self.ids = [i for i in self.vehicle_controllers]

        # (could cause error, port occupied, should catch for exception)
        # TODO: Catch sumo/traci errors
        # TODO: Expand for start time, end time, step length

        logging.info(" Starting SUMO on port " + str(self.port) + "!")
        logging.debug(" Cfg file " +  str(self.cfg))
        self.sumoProcess = subprocess.Popen([self.sumo_binary, "-c", self.cfg, "--remote-port",
                                        str(self.port), "--step-length", str(0.1)], stdout=sys.stdout, stderr=sys.stderr)

        logging.info(" Initializing TraCI on port " + str(self.port) + "!")
        traci.init(self.port)
        for car_id in self.ids:
            # IMPORTANT FOR FINE GRAIN CONTROL OF VEHICLE SPEED
            traci.vehicle.setSpeedMode(car_id, 0)

        logging.info("first step")
        traci.simulationStep()
        logging.info("first step complete")

        for index, car_id in enumerate(self.rl_ids):
            logging.info("Car with id " + car_id + " is on route " + str(traci.vehicle.getRouteID(car_id)))
            logging.info("Car with id " + car_id + " is on edge " + str(traci.vehicle.getLaneID(car_id)))
            logging.info("Car with id " + car_id + " has valid route: " + str(traci.vehicle.isRouteValid(car_id)))
            logging.info("Car with id " + car_id + " has speed: " + str(traci.vehicle.getSpeed(car_id)))
            logging.info("Car with id " + car_id + " has pos: " + str(traci.vehicle.getPosition(car_id)))
            logging.info("Car with id " + car_id + " has route: " + str(traci.vehicle.getRoute(car_id)))
            logging.info("Car with id " + car_id + " is at index: " + str(traci.vehicle.getRouteIndex(car_id)))

        # may have to manually add cars here
        self.initialize_simulation()


    def initialize_simulation(self):
        """
        Needs to generate, and place cars in the correct places, will use information from inital_config

        (this method can be overridden, so init never needs to be)
        """
        raise NotImplementedError

    def apply_action(self, car_id, action):
        """
        :param car_id:
        :param action: as specified by a controller or RL lab, usually a scalar value
        """
        raise NotImplementedError

    def step(self, rl_actions):
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
        logging.info("================= performing step =================")
        for car_id in self.controlled_ids:
            action = self.vehicle_controllers[car_id].get_action()
            self.apply_action(car_id, action=action)
            logging.info("Car with id " + car_id + " is on route " + str(traci.vehicle.getRouteID(car_id)))

        for index, car_id in enumerate(self.rl_ids):
            action = rl_actions[index]
            self.apply_action(car_id, action=action)

        traci.simulationStep()
        logging.info(traci.simulation.getArrivedIDList())

        for index, car_id in enumerate(self.rl_ids):
            logging.info("Car with id " + car_id + " is on route " + str(traci.vehicle.getRouteID(car_id)))
            logging.info("Car with id " + car_id + " is on edge " + str(traci.vehicle.getLaneID(car_id)))
            logging.info("Car with id " + car_id + " has valid route: " + str(traci.vehicle.isRouteValid(car_id)))
            logging.info("Car with id " + car_id + " has speed: " + str(traci.vehicle.getSpeed(car_id)))
            logging.info("Car with id " + car_id + " has pos: " + str(traci.vehicle.getPosition(car_id)))
            logging.info("Car with id " + car_id + " has route: " + str(traci.vehicle.getRoute(car_id)))
            logging.info("Car with id " + car_id + " is at index: " + str(traci.vehicle.getRouteIndex(car_id)))

        self._state = np.array([traci.vehicle.getSpeed(vID) for vID in self.controlled_ids])
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
        raise NotImplementedError

    @property
    def action_space(self):
        """
        Returns an action space object
        """
        raise NotImplementedError
    @property
    def observation_space(self):
        """
        Returns an observation space object
        """
        raise NotImplementedError

    def compute_reward(self, state):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def terminate(self):
        traci.close()



