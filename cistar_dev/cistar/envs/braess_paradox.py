from cistar.envs.base_env import SumoEnvironment
from cistar.core import rewards

from rllab.spaces import Box
from rllab.spaces import Product

import numpy as np
from numpy.random import normal
import random

import pdb


class BraessParadoxEnvironment(SumoEnvironment):
    """
    A class meant to mimic Braess's paradox.

    Human vehicles vehicles make route choices that maximize their travel time, and continuously adjust
    their perception of the travel time through the available routes.

    RL vehicles are also allowed to make route choices; however, these actions are not set by the expected
    travel time. In addition, RL vehicle may slow down / stop.
    """
    def __init__(self, env_params, sumo_binary, sumo_params, scenario):
        """
        See parent class
        """
        super().__init__(env_params, sumo_binary, sumo_params, scenario)

        # specifies whether vehicles are allowed to cross the edge connecting
        # the top and bottom portions of the network
        self.close_CD = self.env_params["close_CD"]

        # the route choice variable contacts the edges to traverse for each of the
        # braess paradox route choices
        if self.close_CD:
            self.available_route_choices = [["BA2", "AC", "CB"],
                                            ["BA2", "AD", "DB"]]
        else:
            self.available_route_choices = [["BA2", "AC", "CD", "DB"],
                                            ["BA2", "AC", "CB"],
                                            ["BA2", "AD", "DB"]]

        # The expected travel time for each of the 3 routes is the average travel time
        # a vehicle experienced during the rollout in each route. Each human vehicles
        # chooses the route that minimizes its expected travel time. Among routes with
        # similar expected travel times, the driver chooses routes at random.
        self.current_braess_route_choice = dict()
        self.route_expected_tt = dict()
        self.current_route_times = dict()
        for veh_id in self.ids:
            self.current_braess_route_choice[veh_id] = []
            self.route_expected_tt[veh_id] = [(0, 0)] * len(self.available_route_choices)
            self.current_route_times[veh_id] = dict()
            self.current_route_times[veh_id]["enter"] = [0, 0]
            self.current_route_times[veh_id]["exit"] = [0, 0]

        # initialize edge memory (used in setting speed)
        self.prev_edge = dict()
        self.current_edge = dict()
        for veh_id in self.ids:
            self.prev_edge[veh_id] = self.vehicles.get_edge(veh_id)
            self.current_edge[veh_id] = self.vehicles.get_edge(veh_id)

    @property
    def action_space(self):
        """
        See parent class
        Moves consist of routing decisions, as well as accelerations performed by rl vehicles.
        """
        lb = [0, - np.abs(self.env_params["max-deacc"])] * self.vehicles.num_rl_vehicles
        ub = [2, self.env_params["max-acc"]] * self.vehicles.num_rl_vehicles

        return Box(np.array(lb), np.array(ub))

    @property
    def observation_space(self):
        """
        See parent class
        """
        speed = Box(low=0, high=np.inf, shape=(self.vehicles.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))

        return Product([speed, absolute_pos])

    def apply_rl_actions(self, rl_actions):
        """
        See parent class
        """
        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]

        # the variable route_num contains the index of the route rl vehicles would like to perform,
        # as presented in self.available_route_choices in __init__
        route_num = np.round(rl_actions[::2])

        rl_routes = [self.available_route_choices[int(route_i)] for route_i in route_num]
        rl_acc = rl_actions[1::2]

        self.apply_acceleration(sorted_rl_ids, rl_acc)

        self.choose_routes(sorted_rl_ids, rl_routes)

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        return rewards.desired_velocity(self, fail=kwargs["fail"])

    def get_state(self):
        """
        See parent class
        """
        return np.array([[self.vehicles.get_speed(vehicle) + normal(0, self.observation_vel_std),
                          self.vehicles.get_absolute_position(vehicle) + normal(0, self.observation_pos_std)]
                         for vehicle in self.sorted_ids]).T

    def step(self, rl_actions):
        """
        See parent class
        Prior to performing base_env's step function, vehicles are allowed to enter the network
        if requested, and the lists of vehicle id's are updated.
        """
        # store the previous edge for each vehicle
        for veh_id in self.ids:
            self.current_edge[veh_id] = self.vehicles.get_edge(veh_id)

        # in addition to regular actions, vehicles in the braess paradox scenario can
        # can also choose routes
        self.choose_routes(self.controlled_ids)

        # continue with performing requested actions and updating the observation space
        output = super().step(rl_actions)

        for veh_id in self.ids:
            # update previous edge data
            self.prev_edge[veh_id] = self.current_edge[veh_id]

            current_edge = self.vehicles.get_edge(veh_id)

            # update the vehicle entrance and exit times
            if current_edge in ["AC", "AD"] and self.prev_edge[veh_id] not in ["AC", "AD"]:
                self.current_route_times[veh_id]["enter"] = self.timer
                self.current_braess_route_choice[veh_id] = [self.vehicles.get_edge(veh_id)]

            elif current_edge == "CD" and self.prev_edge[veh_id] != "CD":
                self.current_braess_route_choice[veh_id].append(self.vehicles.get_edge(veh_id))

            elif current_edge in ["CB", "DB"] and self.prev_edge[veh_id] not in ["CB", "DB"]:
                self.current_braess_route_choice[veh_id].append(self.vehicles.get_edge(veh_id))
                # self.current_route_times[veh_id]["exit"][0] = self.timer
                # if self.vehicles[veh_id]["edge"] in ["CB", "D"]:
                #     self.current_route_times[veh_id]["enter"][1] = self.timer

            elif current_edge == "B" and self.prev_edge[veh_id] != "B":
                self.current_route_times[veh_id]["exit"] = self.timer

                # if the vehicle exited the braess paradox network, calculate the amount of
                # time it needed to traverse the network
                try:
                    current_route_indx = np.where([set(x[1:]) == set(self.current_braess_route_choice[veh_id])
                                                   for x in self.available_route_choices])[0][0]
                except IndexError:
                    # this exception is needed for vehicles at the beginning of the run, before they have
                    # made a full run through the braess' section of the network
                    break

                # update the driver's perception of the expected time needed to traverse the network
                # given a specific route
                num_attempts, expected_tt = self.route_expected_tt[veh_id][current_route_indx]
                enter_time_steps = self.current_route_times[veh_id]["enter"]
                exit_time_steps = self.current_route_times[veh_id]["exit"]

                new_tt = (exit_time_steps - enter_time_steps) * self.time_step

                new_num_attempts = num_attempts + 1
                new_expected_tt = (num_attempts * expected_tt + new_tt) / (num_attempts + 1)

                self.route_expected_tt[veh_id][current_route_indx] = (new_num_attempts, new_expected_tt)

                print(self.route_expected_tt[veh_id])

        return output

    def reset(self):
        """
        See parent class
        """
        observation = super().reset()

        for veh_id in self.ids:
            # reset edge memory (used in setting speed)
            self.prev_edge[veh_id] = self.vehicles.get_edge(veh_id)
            self.current_edge[veh_id] = self.vehicles.get_edge(veh_id)

            # reset memory on expected travel time in routes for vehicles in the network
            self.route_expected_tt[veh_id] = [(0, 0)] * len(self.available_route_choices)

        return observation

    def choose_routes(self, veh_ids, route_choices=None):
        """
        The vehicle is close enough to the regions in which rerouting decisions are applicable (i.e. nodes A and C,
        as presented in braess_paradox_scenario), the vehicle is allowed to make based on a user-specified input,
        or by perceiving the options

        :param veh_ids: list of vehicle identifiers
        :param route_choices: list of edges the vehicle would like to traverse through.
                              If none is given, the vehicle will choose the route that will allow it to move fastest
        """
        # used to check if decisions on the vehicle's route are being made by the rl agent or not
        if route_choices is None:
            route_choices = [None] * len(veh_ids)

        for i, veh_id in enumerate(veh_ids):
            # some values of interest
            this_edge = self.vehicles.get_edge(veh_id)

            # if the vehicle is not at the end of its route, do not make any route changing decisions
            if this_edge != self.vehicles.get_route(veh_id)[-1]:
                continue

            # in order for vehicles to remain in the network indefinitely, they reroute once they are near node "B"
            if this_edge in ["DB", "CB"]:
                route_choices[i] = [this_edge, "B", "BA1", "BA2"]

            # if no route is given, and the vehicle is not in a position to reroute back into the braess network,
            # choose the route based on which one will allow the car to move fastest
            if route_choices[i] is None:
                # expected travel time of each route for the vehicle
                tt = np.array([j[1] for j in self.route_expected_tt[veh_id]])

                if self.close_CD:
                    current_route_choice_indx = np.argmin(tt)
                    # current_route_choice_indx = random.choice([0, 2])
                else:
                    # the vehicle chooses the route that will allow it to move fastest
                    # choices between equivalent routes are made randomly
                    # current_route_choice_indx = random.choice(np.where(min(tt) == tt)[0])
                    current_route_choice_indx = np.argmin(tt)

                route_choices[i] = self.available_route_choices[current_route_choice_indx]

            if self.vehicles.get_route(veh_id) != route_choices[i]:
                self.vehicles.set_route(veh_id, route_choices[i])
                self.traci_connection.vehicle.setRoute(vehID=veh_id, edgeList=route_choices[i])

    def apply_acceleration(self, veh_ids, acc):
        """
        See parent class
        """
        i_called = []  # index of vehicles in the list that have already received traci calls
        for i, veh_id in enumerate(veh_ids):
            this_edge = self.vehicles.get_edge(veh_id)
            this_lane = self.vehicles.get_lane(veh_id)

            # vehicles on any of the edges that are not part of the braess paradox network
            # move at the lower speed limit - this should serve to mimic constant speeds in the lower
            # speed limit lanes, while limiting accelerations by density in the higher speed limit lanes,
            # thereby making velocity in these lanes a function of density
            if this_edge in ["B", "BA1", "BA2"]:
                target_speed = min(self.scenario.net_params["AC_DB_speed_limit"],
                                   self.scenario.net_params["AD_CB_speed_limit"])

                # in order to ensure that vehicles are no adjacent to one another along the connecting route, thereby
                # blocking each other from switching lanes, all cars are placed at the center lane, and vehicles not
                # on the lane move at a slower speed so that they are not always next to the vehicles.
                if this_edge in ["BA1"] and this_lane != 1:
                    target_speed = max(0, target_speed - 10)
                    self.traci_connection.vehicle.changeLane(veh_id, 1, 100000)

                self.traci_connection.vehicle.slowDown(vehID=veh_id, speed=target_speed, duration=1)
                i_called.append(i)

        # delete data on vehicles that have already received traci calls in order to reduce run-time
        veh_ids = [veh_ids[i] for i in range(len(veh_ids)) if i not in i_called]
        acc = [acc[i] for i in range(len(acc)) if i not in i_called]

        super().apply_acceleration(veh_ids=veh_ids, acc=acc)

    def additional_command(self):
        """
        See parent class
        In order to mimic variable speed limits, the desired speed limit on the IDMController (v0) is modified
        depending on the lane the vehicle is located.
        Lane changes in braess are used to keep vehicles in the lane designated to their specified routes.
        """
        for veh_id in self.controlled_ids:
            this_edge = self.vehicles.get_edge(veh_id)
            this_lane = self.vehicles.get_lane(veh_id)
            this_route_choice = set(self.vehicles.get_route(veh_id))

            target_lane = None

            if this_edge == "BA2":
                if this_route_choice == {"BA2", "AC", "CB"} and this_lane != 2:
                    target_lane = 2
                elif this_route_choice == {"BA2", "AC", "CD", "DB"} and this_lane != 1:
                    target_lane = 1
                elif this_route_choice == {"BA2", "AD", "DB"} and this_lane != 0:
                    target_lane = 0

            elif this_edge == "AC":
                if this_route_choice == {"BA2", "AC", "CB"} and this_lane != 1:
                    target_lane = 1
                elif this_route_choice == {"BA2", "AC", "CD", "DB"} and this_lane != 0:
                    target_lane = 0

            elif this_edge == "DB":
                if set(self.current_braess_route_choice[veh_id]) == {"AD", "DB"} and this_lane != 0:
                    target_lane = 0
                elif set(self.current_braess_route_choice[veh_id]) == {"AC", "CD", "DB"} and this_lane != 1:
                    target_lane = 1

            if target_lane is not None:
                self.traci_connection.vehicle.changeLane(veh_id, int(target_lane), 100000)

        for veh_id in self.controlled_ids:
            current_edge = self.vehicles[veh_id]["edge"]
            # previous_edge = self.vehicles[veh_id]["previous_edge"]

            if current_edge in ["AD", "CB"] and self.prev_edge[veh_id] not in ["AD", "CB"]:
                self.traci_connection.vehicle.setMaxSpeed(veh_id, self.scenario.net_params["AD_CB_speed_limit"])
                # self.vehicles[veh_id]["controller"].v0 = self.scenario.net_params["AD_CB_speed_limit"]

            elif current_edge in ["AC", "DB"] and self.prev_edge[veh_id] not in ["AC", "CD", "DB"]:
                self.traci_connection.vehicle.setMaxSpeed(veh_id, self.scenario.net_params["AC_DB_speed_limit"])
                # self.vehicles[veh_id]["controller"].v0 = self.scenario.net_params["AC_DB_speed_limit"]
