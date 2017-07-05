"""
This class is an extension the SumoEnvironment class located in base_env.

With this class, vehicles can enter the network stochastically, and exit as soon as
they reach their destination.

(describe how the action and observation spaces are modified)
"""

from cistar.core.base_env import SumoEnvironment

from rllab.spaces import Box
from rllab.spaces import Product

import numpy as np
from numpy.random import normal, uniform
from random import randint

import pdb


class SimpleIntersectionEnvironment(SumoEnvironment):
    """
    Fully functional environment for intersections.
    Vehicles enter the system following a user-specified model.
    The type of each entering vehicle is based on user-specified probability values
    """

    def __init__(self, env_params, sumo_binary, sumo_params, scenario):
        """
        See parent class
        Probability data on entering cars are also added
        """
        # prob_enter contains the probability model used to determine when cars enter
        # this model is (possibly) a function of the time spent since the last car entered
        self.prob_enter = dict()
        self.last_enter_time = dict()
        for key in scenario.net_params["prob_enter"].keys():
            self.prob_enter[key] = scenario.net_params["prob_enter"][key]
            self.last_enter_time[key] = 0

        # prob_vType specifies the probability of each type of car entering the system
        self.vType = scenario.type_params.keys()
        self.prob_vType = np.array([scenario.type_params[key][0] for key in self.vType])
        self.prob_vType = self.prob_vType / sum(self.prob_vType)
        self.prob_vType = np.cumsum(self.prob_vType)

        # the entering speed of each car is set to the max speed of the network
        self.enter_speed = scenario.net_params["speed_limit"]

        super().__init__(env_params, sumo_binary, sumo_params, scenario)

    def step(self, rl_actions):
        """
        See parent class
        Prior to performing base_env's step function, vehicles are allowed to enter the network
        if requested, and the lists of vehicle id's are updated.
        """
        new_vehicles = False
        new_ids = []

        for enter_lane in self.prob_enter.keys():
            x = self.timer + 1 - self.last_enter_time[enter_lane]

            # check if a vehicle wants to enter a lane
            if self.prob_enter[enter_lane](x) > uniform(0, 1):
                new_vehicles = True
                self.last_enter_time[enter_lane] = self.timer + 1

                # if a car wants to enter, determine which type it is
                vType_choice = uniform(0, 1)
                for i in range(len(self.prob_vType)):
                    if vType_choice >= self.prob_vType:
                        new_type_id = self.vType[i]
                        new_veh_id = self.vType[i] + '_'  # TODO: fix me!
                        new_ids.append(new_veh_id)
                        break
                new_lane_index = randint(0, self.scenario.lanes[enter_lane])
                new_route_id = self.scenario.generator.rts[enter_lane]

                # setup the initial conditions of the vehicle and add them to the self.vehicle variable
                self.vehicles[new_veh_id] = 0  # TODO: fix me!

                # add the car to the start of the lane
                self.traci_connection.vehicle.addFull(
                    new_veh_id, new_route_id, typeID=str(new_type_id), departLane=str(new_lane_index),
                    departPos=str(0), departSpeed=str(self.enter_speed))

        # continue with performing requested actions and updating the observation space
        super().step(rl_actions)

        # update the lists of vehicle ids
        if new_vehicles:
            for veh_id in new_ids:
                print("yay!")  # TODO: fix me!

    @property
    def action_space(self):
        """
        See parent class
        """
        # TODO: can these boxes be redefined in between steps?

        # if the network contains only one lane, then the actions are a set of accelerations from max-deacc to max-acc
        if self.scenario.lanes == 1:
            return Box(low=-np.abs(self.env_params["max-deacc"]), high=self.env_params["max-acc"],
                       shape=(self.scenario.num_rl_vehicles, ))

        # if the network contains two or more lanes, the actions also include a lane-changing component
        else:
            lb = [-abs(self.env_params["max-deacc"]), -1] * self.scenario.num_rl_vehicles
            ub = [self.env_params["max-acc"], 1] * self.scenario.num_rl_vehicles
            return Box(np.array(lb), np.array(ub))

    @property
    def observation_space(self):
        """
        See parent class
        """
        speed = Box(low=0, high=np.inf, shape=(self.scenario.num_vehicles,))
        lane = Box(low=0, high=self.scenario.lanes-1, shape=(self.scenario.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))

        if self.scenario.lanes == 1:
            return Product([speed, absolute_pos])
        else:
            return Product([speed, lane, absolute_pos])

    def apply_rl_actions(self, rl_actions):
        """
        See parent class
        """
        sorted_indx = np.argsort([self.vehicles[veh_id]["absolute_position"] for veh_id in self.rl_ids])
        sorted_rl_ids = np.array(self.rl_ids)[sorted_indx]

        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        vel = state[0]

        if any(vel < -100) or kwargs["fail"]:
            return 0.0

        max_cost = np.array([self.env_params["target_velocity"]]*self.scenario.num_vehicles)
        max_cost = np.linalg.norm(max_cost)

        cost = vel - self.env_params["target_velocity"]
        cost = np.linalg.norm(cost)

        return max(max_cost - cost, 0)

    def getState(self, **kwargs):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        sorted_indx = np.argsort([self.vehicles[veh_id]["absolute_position"] for veh_id in self.ids])
        sorted_ids = np.array(self.ids)[sorted_indx]

        return np.array([[self.vehicles[vehicle]["speed"] + normal(0, kwargs["observation_vel_std"]),
                          self.vehicles[vehicle]["absolute_position"] + normal(0, kwargs["observation_pos_std"])]
                         for vehicle in sorted_ids]).T

    def render(self):
        print('current state/velocity:', self.state)
