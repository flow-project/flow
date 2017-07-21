"""
This class is an extension the SumoEnvironment class located in base_env.

It is meant to mimic the traffic network known as Braess's paradox.
This is done by allowing human vehicles to make route choices to make route choices that would maximize their
travel time (i.e. speed).

RL vehicles are also allowed to make route choices, with the hopes that some route choices made by these
vehicles will allow the system to push past Nash Equilibrium and into the Social Optimum.
"""

from cistar.core.base_env import SumoEnvironment
from cistar.envs.loop import LoopEnvironment
from cistar.controllers.base_controller import SumoController
from cistar.controllers.rlcontroller import RLController

from rllab.spaces import Box
from rllab.spaces import Product

import numpy as np
from numpy.random import normal, uniform
from random import randint

import pdb


class BraessParadoxEnvironment(LoopEnvironment):
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
        super().__init__(env_params, sumo_binary, sumo_params, scenario)

        # add necessary components for vehicles to perform routing decisions
        self.rts = self.scenario.generator.rts
        for key in self.rts.keys():
            self.rts[key] = self.rts[key].split(' ')

        # TODO: tune me!
        self.constant_edge_speed = 4.
        self.varying_edge_speed = lambda x: 0.5 / x

    def step(self, rl_actions):
        """
        See parent class
        Prior to performing base_env's step function, vehicles are allowed to enter the network
        if requested, and the lists of vehicle id's are updated.
        """
        for veh_id in self.controlled_ids:
            self.choose_route(veh_id)

        # continue with performing requested actions and updating the observation space
        super().step(rl_actions)

    @property
    def action_space(self):
        """
        See parent class
        """
        # moves consist solely of routing decisions
        # TODO: in mixed autonomy, we may want to give automated vehicles the option of slowing down/stopping
        route_choice = Box(low=0., high=1, shape=(self.scenario.num_vehicles,))
        stop_go = Box(low=0., high=1, shape=(self.scenario.num_vehicles,))

        return Product([route_choice, stop_go])

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
        return
        # # re-arrange actions according to mapping in observation space
        # sorted_rl_ids = np.array([])
        #
        # if all([self.scenario.lanes[key] == 1 for key in self.scenario.lanes]):
        #     acceleration = rl_actions
        #     self.apply_acceleration(sorted_rl_ids, acc=acceleration)
        #
        # else:
        #     acceleration = rl_actions[::2]
        #     direction = np.round(rl_actions[1::2])
        #
        #     # represents vehicles that are allowed to change lanes
        #     non_lane_changing_veh = [self.timer <= self.lane_change_duration + self.vehicles[veh_id]['last_lc']
        #                              for veh_id in sorted_rl_ids]
        #     # vehicle that are not allowed to change have their directions set to 0
        #     direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))
        #
        #     self.apply_acceleration(sorted_rl_ids, acc=acceleration)
        #     self.apply_lane_change(sorted_rl_ids, direction=direction)

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        try:
            vel = state[0]
        except IndexError:
            # if there are no vehicles are in the network, return a fixed reward
            return 0.0  # TODO: should we reward this positively?

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
        """
        return np.array([[self.vehicles[vehicle]["speed"] + normal(0, kwargs["observation_vel_std"]),
                          self.vehicles[vehicle]["absolute_position"] + normal(0, kwargs["observation_pos_std"])]
                         for vehicle in self.sorted_ids]).T

    def render(self):
        print('current state/velocity:', self.state)

    def get_x_by_id(self, id):
        """
        Returns the position of the vehicle with specified id
        :param id: id of vehicle
        :return:
        """
        if self.vehicles[id]["edge"] == '':
            # print("This vehicle teleported and its edge is now empty", id)
            return 0.
        return self.scenario.get_x(self.vehicles[id]["edge"], self.vehicles[id]["position"])

    def choose_route(self, veh_id, route_choice=None):
        """
        The vehicle is close enough to the regions in which rerouting decisions are applicable (i.e. nodes A and C,
        as presented in braess_paradox_scenario), the vehicle is allowed to make based on a user-specified input,
        or by perceiving the options

        :param veh_id: vehicle identifier
        :param route_choice: list of edges the vehicle would like to traverse through.
                             If none is given, the vehicle will choose the route that will allow it to move fastest
        """
        # some constant values of interest
        this_edge = self.vehicles[veh_id]["edge"]
        this_pos = self.vehicles[veh_id]["position"]
        edge_len = self.scenario.edge_len

        # in order for vehicles to remain in the network indefinitely, they reroute once they are near node "B"
        if this_pos >= edge_len - 5 and this_edge in ["D", "CB"]:
            route_choice = "B"

        # this term determines if the vehicle in question is close enough to any route-choosing nodes to perform a
        # a rerouting decision
        near_route_choosing_node = this_pos >= edge_len - 5 and this_edge in ["BA2", "AC"]

        # if the vehicle is not close enough, do not make any route changing decisions
        if not near_route_choosing_node:
            return

        # if no route is given, choose the route based on which one will allow the car to move fastest
        if route_choice is None:
            if this_edge == "AC":
                competing_edge = "D"
            else:  # this_edge == "BA2":
                competing_edge = "AC"

            # density in density-dependant edge
            num_cars = sum([self.vehicles[vID]["edge"] == competing_edge for vID in self.ids]) + 1
            density = num_cars / self.scenario.edge_len

            if this_edge == "AC" and self.constant_edge_speed > self.varying_edge_speed(density):
                route_choice = "CB"
            elif this_edge == "AC" and self.constant_edge_speed <= self.varying_edge_speed(density):
                route_choice = "CD"
            elif this_edge == "BA2" and self.constant_edge_speed > self.varying_edge_speed(density):
                route_choice = "AD"
            elif this_edge == "BA2" and self.constant_edge_speed <= self.varying_edge_speed(density):
                route_choice = "AC"

        # the route traci is given must start with the vehicles current position
        out_route = np.append([self.vehicles[veh_id]['edge']], self.rts[route_choice])

        self.traci_connection.vehicle.setRoute(vehID=veh_id, edgeList=out_route)

    def apply_acceleration(self, veh_ids, acc=None):
        """
        See parent class
        """
        for veh_id in veh_ids:
            # vehicles on edges AC and DB move at a speed dependent on the density
            if self.vehicles[veh_id]["edge"] in ["AC", "D"]:
                # compute the density of cars in the edge the vehicle is currently located
                this_edge = self.vehicles[veh_id]["edge"]
                num_cars = sum([self.vehicles[vID]["edge"] == this_edge for vID in self.ids])
                density = num_cars / self.scenario.edge_len

                self.traci_connection.vehicle.slowDown(veh_id, self.varying_edge_speed(density), 1)

            # vehicles on edges AD and CB moves at a speed independent of the density
            if self.vehicles[veh_id]["edge"] in ["AD", "CB"]:
                self.traci_connection.vehicle.slowDown(veh_id, self.constant_edge_speed, 1)

            # vehicles on edges CD and any of the edges not part of the braess paradox network
            # move very fast (mimicking a zero or negligible travel time)
            if self.vehicles[veh_id]["edge"] in ["CD", "B", "BA1", "BA2"]:
                fast_speed = 30.
                self.traci_connection.vehicle.slowDown(veh_id, fast_speed, 1)
