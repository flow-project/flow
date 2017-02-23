import logging

from rllab.envs.base import Env
from rllab.envs.base import Step

import numpy as np

import subprocess, sys

import copy
import traci


"""
This file provides the interface for controlling a SUMO simulation. Using the environment class, you can
start sumo, provide a scenario to specify a configuration and controllers, perform simulation steps, and
reset the simulation to an initial configuration.

This class cannot be used as is, as you must extend it to implement an action applicator method, and
properties to define the MDP if you choose to use it with RLLab.

"""


class SumoEnvironment(Env):

    def __init__(self,env_params, sumo_binary, sumo_params, scenario):
        """
        Initialize the Sumo Environment, by starting SUMO, setting up TraCI and initializing vehicles
        Input
        -----
        env_params   : use this dictionary to pass in parameters relevant to the environment
                     (i.e. target velocities, constants for step function)
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
        self.type_params = scenario.type_params
        self.sumo_binary = sumo_binary
        self.scenario = scenario
        self.initial_state = {}
        self.vehicles = {}
        self.timer = 0

        if "port" not in sumo_params:
            raise ValueError("SUMO port not defined")
        else:
            self.port = sumo_params["port"]

        self.cfg = scenario.cfg

        logging.info(" Starting SUMO on port " + str(self.port) + "!")
        logging.debug(" Cfg file " +  str(self.cfg))

        self.time_step = 0.01
        if "time_step" in sumo_params:
            self.time_step = sumo_params["time_step"]

        subprocess.Popen([self.sumo_binary, "-c", self.cfg, "--remote-port",
                                        str(self.port), "--step-length", str(self.time_step)], stdout=sys.stdout, stderr=sys.stderr)

        logging.debug(" Initializing TraCI on port " + str(self.port) + "!")
        traci.init(self.port)

        self.initialize_simulation()

        self.ids = traci.vehicle.getIDList()
        self.controlled_ids = [i for i in self.ids if not traci.vehicle.getTypeID(i) == "rl"]
        self.rl_ids = [i for i in self.ids if traci.vehicle.getTypeID(i) == "rl"]


        for index, car_id in enumerate(self.ids):
            vehicle = {}
            vehicle["id"] = car_id
            vehicle["type"] = traci.vehicle.getTypeID(car_id)
            vehicle["edge"] = traci.vehicle.getRoadID(car_id)
            vehicle["position"] = traci.vehicle.getLanePosition(car_id)
            vehicle["lane"] = traci.vehicle.getLaneIndex(car_id)
            vehicle["speed"] = traci.vehicle.getSpeed(car_id)
            vehicle["length"] = traci.vehicle.getLength(car_id)
            vehicle["max_speed"] = traci.vehicle.getMaxSpeed(car_id)
            self.vehicles[car_id] = vehicle
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
        while traci.vehicle.getIDCount() < self.num_vehicles:
            traci.simulationStep()
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
            action = self.type_params[car_type][1](car_id, self)
            self.apply_action(car_id, action=action)
            logging.debug("Car with id " + car_id + " is on route " + str(traci.vehicle.getRouteID(car_id)))

        for index, car_id in enumerate(self.rl_ids):
            action = rl_actions[index]
            self.apply_action(car_id, action=action)

        traci.simulationStep()
        self.timer += 1
        # if it's been long enough
        # try and change lanes
        if self.timer == 80:
            print(' ')
            print(' ')
            self.timer = 0
            for car_id in self.controlled_ids:
                car_type = traci.vehicle.getTypeID(car_id)
                self.type_params[car_type][2](car_id, self)

        self.last_step = copy.deepcopy(self.vehicles)

        for index, car_id in enumerate(self.ids):
            self.vehicles[car_id]["type"] = traci.vehicle.getTypeID(car_id)
            self.vehicles[car_id]["edge"] = traci.vehicle.getRoadID(car_id)
            self.vehicles[car_id]["position"] = traci.vehicle.getLanePosition(car_id)
            self.vehicles[car_id]["lane"] = traci.vehicle.getLaneIndex(car_id)
            self.vehicles[car_id]["speed"] = traci.vehicle.getSpeed(car_id)

        self._state = np.array([traci.vehicle.getSpeed(vID) for vID in self.ids])
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

        traci.simulationStep()

        for index, car_id in enumerate(self.ids):
            traci.vehicle.setSpeedMode(car_id, 0)
            self.vehicles[car_id]["type"] = traci.vehicle.getTypeID(car_id)
            self.vehicles[car_id]["edge"] = traci.vehicle.getRoadID(car_id)
            self.vehicles[car_id]["position"] = traci.vehicle.getLanePosition(car_id)
            self.vehicles[car_id]["lane"] = traci.vehicle.getLaneIndex(car_id)
            self.vehicles[car_id]["speed"] = traci.vehicle.getSpeed(car_id)

        self._state = np.array([traci.vehicle.getSpeed(vID) for vID in self.ids])
        observation = np.copy(self._state)
        return observation

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
