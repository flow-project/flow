import logging

from rllab.envs.base import Env
from rllab.envs.base import Step

import numpy as np

import subprocess, sys

import random

import copy
import traci


"""
This file provides the interface for controlling a SUMO simulation. Using the environment class, you can
start sumo, provide a scenario to specify a configuration and controllers, perform simulation steps, and
reset the simulation to an initial configuration.

This class cannot be used as is: you must extend it to implement an action applicator method, and
properties to define the MDP if you choose to use it with RLLab.

"""

COLORS = [(255,0,0,0),(0, 255,0,0),(0, 0, 255,0), (255, 255,0,0),(0, 255,255,0),(255, 0,255,0), (255, 255,255,0)]


class SumoEnvironment(Env):

    def __init__(self,env_params, sumo_binary, sumo_params, scenario):
        """[summary]
        
        [description]
        
        Arguments:
            env_params {dictionary} -- [description]
            sumo_binary {string} -- Either "sumo" or "sumo-gui"
            sumo_params {dictionary} -- {"port": {integer} connection to SUMO,
                            "timestep": {float} default=0.01s, SUMO default=1.0s}
            scenario {Scenario} -- @see Scenario; abstraction of the SUMO XML files
                                    which specify the vehicles placed on the net
        
        Raises:
            ValueError -- Raised if a SUMO port not provided
        """
        self.env_params = env_params
        self.sumo_binary = sumo_binary
        self.scenario = scenario
        
        # Vehicles: Key = Vehicle ID, Value = Dictionary describing the vehicle
        self.vehicles = {}

        # Represents number of steps taken
        self.timer = 0
        # 0.01 = default time step for our research
        self.time_step = sumo_params["time_step"] if "time_step" in sumo_params else 0.01

        if "port" not in sumo_params:
            raise ValueError("SUMO port not defined")

        logging.info(" Starting SUMO on port " + str(sumo_params["port"]))
        logging.debug(" Cfg file " + str(self.scenario.cfg))

        # Opening the I/O thread to SUMO
        subprocess.Popen([self.sumo_binary, "-c", self.scenario.cfg, "--remote-port",
                                        str(sumo_params["port"]), "--step-length", str(self.time_step)], stdout=sys.stdout, stderr=sys.stderr)

        logging.debug(" Initializing TraCI on port " + str(sumo_params["port"]) + "!")
        traci.init(sumo_params["port"])

        self.ids = traci.vehicle.getIDList()

        self.controlled_ids, self.rl_ids = [], []
        for i in self.ids:
            if traci.vehicle.getTypeID(i) == "rl":
                self.rl_ids.append(i)
            else:
                self.controlled_ids.append(i)

        # TODO: could possibly be handled in sumo experiment
        # Initial state: Key = Vehicle ID, Entry = (type_id, route_id, lane_index, lane_pos, speed, pos) 
        self.initial_state = {}
        self.setup_initial_state()

    def setup_initial_state(self):
        """
        Store initial state so that simulation can be reset at the end.
        TODO: Make traci calls as bulk as possible
        Initial state is a dictionary: key = vehicle IDs, value = state describing car
        """
        for veh_id in self.ids:
            vehicle = {}
            vehicle["id"] = veh_id
            vehicle["type"] = traci.vehicle.getTypeID(veh_id)
            vehicle["edge"] = traci.vehicle.getRoadID(veh_id)
            vehicle["position"] = traci.vehicle.getLanePosition(veh_id)
            vehicle["lane"] = traci.vehicle.getLaneIndex(veh_id)
            vehicle["speed"] = traci.vehicle.getSpeed(veh_id)
            vehicle["length"] = traci.vehicle.getLength(veh_id)
            vehicle["max_speed"] = traci.vehicle.getMaxSpeed(veh_id)
            self.vehicles[veh_id] = vehicle
            traci.vehicle.setSpeedMode(veh_id, 0)

            # Saving initial state
            route_id = traci.vehicle.getRouteID(veh_id)
            pos = traci.vehicle.getPosition(veh_id)
            # TODO: Should we save some of the arguments as strings so we don't
            # have to repeatedly call str() in reset()
            self.initial_state[veh_id] = (vehicle["type"], route_id, vehicle["lane"], vehicle["position"], vehicle["speed"], pos)

            logging.debug("Car with id " + veh_id + " is on route " + str(route_id))
            logging.debug("Car with id " + veh_id + " is on edge " + str(traci.vehicle.getLaneID(veh_id)))
            logging.debug("Car with id " + veh_id + " has valid route: " + str(traci.vehicle.isRouteValid(veh_id)))
            logging.debug("Car with id " + veh_id + " has speed: " + str(vehicle["speed"]))
            logging.debug("Car with id " + veh_id + " has absolute position: " + str(pos))
            logging.debug("Car with id " + veh_id + " has route: " + str(traci.vehicle.getRoute(veh_id)))
            logging.debug("Car with id " + veh_id + " is at route index: " + str(traci.vehicle.getRouteIndex(veh_id)))

    def step(self, rl_actions):
        """
        Run one timestep of the environment's dynamics. "Self-driving cars" will
        step forward based on rl_actions, provided by the RL algorithm. Other cars
        will step forward based on their car following model. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        rl_actions : an action provided by the rl algorithm
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        logging.debug("================= performing step =================")
        for veh_id in self.controlled_ids:
            car_type = traci.vehicle.getTypeID(veh_id)
            action = self.scenario.type_params[car_type][1](veh_id, self)
            self.apply_action(veh_id, action=action)
            logging.debug("Car with id " + veh_id + " is on route " + str(traci.vehicle.getRouteID(veh_id)))

        for index, veh_id in enumerate(self.rl_ids):
            action = rl_actions[index]
            self.apply_action(veh_id, action=action)

        self.timer += 1
        # TODO: Turn 100 into a hyperparameter
        # if it's been long enough try and change lanes
        if self.timer % 100 == 0:
            for veh_id in self.controlled_ids:
                car_type = self.vehicles[veh_id]["type"]
                newlane = self.scenario.type_params[car_type][2](veh_id, self)
                traci.vehicle.changeLane(veh_id, newlane, 10000)

        traci.simulationStep()

        for veh_id in self.ids:
            self.vehicles[veh_id]["type"] = traci.vehicle.getTypeID(veh_id)
            self.vehicles[veh_id]["edge"] = traci.vehicle.getRoadID(veh_id)
            self.vehicles[veh_id]["position"] = traci.vehicle.getLanePosition(veh_id)
            self.vehicles[veh_id]["lane"] = traci.vehicle.getLaneIndex(veh_id)
            self.vehicles[veh_id]["speed"] = traci.vehicle.getSpeed(veh_id)

        # TODO: Can self._state be initialized, saved and updated so that we can
        # exploit numpy speed
        self._state = self.getState()
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
        color = random.sample(COLORS, 1)[0]
        for veh_id in self.ids:
            type_id, route_id, lane_index, lane_pos, speed, pos = self.initial_state[veh_id]

            logging.debug("Moving car " + veh_id + " from " + str(traci.vehicle.getPosition(veh_id)) + " to " + str(pos))
            traci.vehicle.remove(veh_id)
            traci.vehicle.addFull(veh_id, route_id, typeID=str(type_id), departLane=str(lane_index),
                                  departPos=str(lane_pos), departSpeed=str(speed))
            traci.vehicle.setColor(veh_id, color)
        traci.simulationStep()

        # TODO: Replace these traci calls with initial_state accesses
        for veh_id in self.ids:
            traci.vehicle.setSpeedMode(veh_id, 0)
            self.vehicles[veh_id]["type"] = traci.vehicle.getTypeID(veh_id)
            self.vehicles[veh_id]["edge"] = traci.vehicle.getRoadID(veh_id)
            self.vehicles[veh_id]["position"] = traci.vehicle.getLanePosition(veh_id)
            self.vehicles[veh_id]["lane"] = traci.vehicle.getLaneIndex(veh_id)
            self.vehicles[veh_id]["speed"] = traci.vehicle.getSpeed(veh_id)

        self._state = np.array([traci.vehicle.getSpeed(vID) for vID in self.ids])
        observation = np.copy(self._state)

        return observation

    def apply_action(self, veh_id, action):
        """
        :param veh_id: {string}
        :param action: as specified by a controller or rllab, usually a scalar value
        """
        raise NotImplementedError

    def getState(self):
        """
        returns the state of the simulation, dependent on the experiment/environment
        :return:
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
        """Reward function for RL.
        
        Arguments:
            state {Array-type} -- State of all the vehicles in the simulation
        """
        raise NotImplementedError

    def render(self):
        """Description of the state for when verbose mode is on."""
        raise NotImplementedError

    def terminate(self):
        """Closes the TraCI I/O connection. Should be done at end of every experiment.
        Must be in Environment because the environment opens the TraCI connection.
        """
        traci.close()
