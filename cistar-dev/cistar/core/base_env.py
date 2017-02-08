import logging

from rllab.envs.base import Env
from rllab.envs.base import Step

import numpy as np

import subprocess, sys

import traci


"""
This file contains methods for sumo++ interactions with rllab
Most notably, it has the SumoEnvironment class, which serves as the environment
that gets passed in to rllab's algorithm (it's basically an MDP)

When defining a SumoEnvironment, one must still implement the action space, 
observation space, reward, etc. (properties of the MDP). This class is meant to 
serve as a parent.

"""


class SumoEnvironment(Env):

    def __init__(self,env_params, type_params, sumo_binary, sumo_params, scenario):
        """
        Initialize the Sumo Environment, by starting SUMO, setting up TraCI and initializing vehicles
        Input
        -----
        num_vehicles : total number of vehicles, RL and controller based
        env_params   : use this dictionary to pass in parameters relevant to the environment
                     (i.e. shape, size of simulatiomn; constants for step functions, etc. )
        type_params : dictionary of car -> (count, controller) assignments
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

        self.num_vehicles = scenario.num_vehicles
        self.num_rl_vehicles = scenario.num_rl_vehicles
        self.env_params = env_params
        self.type_params = type_params
        self.sumo_binary = sumo_binary
        self.initial_state = {}

        if "port" not in sumo_params:
            raise ValueError("SUMO port not defined")
        else:
            self.port = sumo_params["port"]

        self.cfg = scenario.cfg

        # (could cause error, port occupied, should catch for exception)
        # TODO: Catch sumo/traci errors
        # TODO: Expand for start time, end time, step length

        logging.info(" Starting SUMO on port " + str(self.port) + "!")
        logging.debug(" Cfg file " +  str(self.cfg))
        subprocess.Popen([self.sumo_binary, "-c", self.cfg, "--remote-port",
                                        str(self.port), "--step-length", str(0.1)], stdout=sys.stdout, stderr=sys.stderr)

        logging.debug(" Initializing TraCI on port " + str(self.port) + "!")
        traci.init(self.port)

        self.initialize_simulation()

        self.ids = traci.vehicle.getIDList()
        self.controlled_ids = [i for i in self.ids if not traci.vehicle.getTypeID(i) == "rl"]
        self.rl_ids = [i for i in self.ids if traci.vehicle.getTypeID(i) == "rl"]

        for index, car_id in enumerate(self.ids):
            traci.vehicle.setSpeedMode(car_id, 0)
            logging.debug("Car with id " + car_id + " is on route " + str(traci.vehicle.getRouteID(car_id)))
            logging.debug("Car with id " + car_id + " is on edge " + str(traci.vehicle.getLaneID(car_id)))
            logging.debug("Car with id " + car_id + " has valid route: " + str(traci.vehicle.isRouteValid(car_id)))
            logging.debug("Car with id " + car_id + " has speed: " + str(traci.vehicle.getSpeed(car_id)))
            logging.debug("Car with id " + car_id + " has pos: " + str(traci.vehicle.getPosition(car_id)))
            logging.debug("Car with id " + car_id + " has route: " + str(traci.vehicle.getRoute(car_id)))
            logging.debug("Car with id " + car_id + " is at index: " + str(traci.vehicle.getRouteIndex(car_id)))

        # could possibly be handled in sumo experiment
        self.store_initial_state()

    def initialize_simulation(self):
        """
        Needs to generate, and place cars in the correct places,
        If cars are placed on routes in CFG, or generated in flows, step until all cars are on the board

        (this method can and should be overridden for different initializations )
        """
        print("there are " +  str(traci.vehicle.getIDCount()) + " cars loaded")
        while traci.vehicle.getIDCount() < self.num_vehicles:
            traci.simulationStep()
            print("there are " + str(traci.vehicle.getIDCount()) + " cars loaded")
            for car_id in traci.vehicle.getIDList():
                # IMPORTANT FOR FINE GRAIN CONTROL OF VEHICLE SPEED
                traci.vehicle.setSpeedMode(car_id, 0)
                traci.vehicle.setSpeed(car_id, 10)


    def store_initial_state(self):
        """
        Store initial state so that simulation can be reset at the end
        """

        # Get initial state of each vehicle, and store in initial config object
        for veh_id in self.ids:
            type_id = traci.vehicle.getTypeID(veh_id)
            route_id = traci.vehicle.getRouteID(veh_id)
            lane_index = traci.vehicle.getLaneIndex(veh_id)
            lane_pos = traci.vehicle.getLanePosition(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            pos = traci.vehicle.getPosition(veh_id)
            self.initial_state[veh_id] = (type_id, route_id, lane_index, lane_pos, speed, pos)

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
        logging.debug("================= performing step =================")
        for car_id in self.controlled_ids:
            car_type = traci.vehicle.getTypeID(car_id)
            action = self.type_params[car_type][1]()
            self.apply_action(car_id, action=action)
            logging.debug("Car with id " + car_id + " is on route " + str(traci.vehicle.getRouteID(car_id)))

        for index, car_id in enumerate(self.rl_ids):
            action = rl_actions[index]
            self.apply_action(car_id, action=action)

        traci.simulationStep()

        for index, car_id in enumerate(self.rl_ids):
            logging.debug("Car with id " + car_id + " is on route " + str(traci.vehicle.getRouteID(car_id)))
            logging.debug("Car with id " + car_id + " is on edge " + str(traci.vehicle.getLaneID(car_id)))
            logging.debug("Car with id " + car_id + " has valid route: " + str(traci.vehicle.isRouteValid(car_id)))
            logging.debug("Car with id " + car_id + " has speed: " + str(traci.vehicle.getSpeed(car_id)))
            logging.debug("Car with id " + car_id + " has pos: " + str(traci.vehicle.getPosition(car_id)))
            logging.debug("Car with id " + car_id + " has route: " + str(traci.vehicle.getRoute(car_id)))
            logging.debug("Car with id " + car_id + " is at index: " + str(traci.vehicle.getRouteIndex(car_id)))

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

        for car_id in self.ids:
            type_id, route_id, lane_index, lane_pos, speed, pos = self.initial_state[car_id]

            logging.debug("Moving car " + car_id + " from " + str(traci.vehicle.getPosition(car_id)) + " to " + str(pos))
            traci.vehicle.remove(car_id)
            traci.vehicle.addFull(car_id, route_id, typeID=str(type_id), departLane=str(lane_index),
                                  departPos=str(lane_pos), departSpeed=str(speed))
            traci.vehicle.setSpeedMode(car_id, 0)

        traci.simulationStep()

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


