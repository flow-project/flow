from cistar.envs.loop import LoopEnvironment
from cistar.core import rewards

from rllab.spaces import Box
from rllab.spaces import Product

import numpy as np
from numpy.random import normal
import random

import pdb


class BraessParadoxEnvironment(LoopEnvironment):
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

        # the route choice variable contacts the edges to traverse for each of the
        # braess paradox route choices
        self.available_route_choices = [["BA2", "AC", "CB"],
                                        ["BA2", "AC", "CD", "DB"],
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

        # specifies whether vehicles are allowed to cross the edge connecting
        # the top and bottom portions of the network
        self.close_CD = self.env_params["close_CD"]

        # initialize edge memory (used in setting speed)
        self.prev_edge = dict()
        self.current_edge = dict()
        for veh_id in self.ids:
            self.prev_edge[veh_id] = self.vehicles[veh_id]["edge"]
            self.current_edge[veh_id] = self.vehicles[veh_id]["edge"]

    @property
    def action_space(self):
        """
        See parent class
        Moves consist of routing decisions, as well as accelerations performed by rl vehicles.
        """
        lb = [0, - np.abs(self.env_params["max-deacc"])] * self.scenario.num_rl_vehicles
        ub = [2, self.env_params["max-acc"]] * self.scenario.num_rl_vehicles

        return Box(np.array(lb), np.array(ub))

    @property
    def observation_space(self):
        """
        See parent class
        """
        speed = Box(low=0, high=np.inf, shape=(self.scenario.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))

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
        reward = rewards.desired_velocity(
            state, rl_actions, fail=kwargs["fail"], target_velocity=self.env_params["target_velocity"])

        return reward

    def getState(self):
        """
        See parent class
        """
        return np.array([[self.vehicles[vehicle]["speed"] + normal(0, self.observation_vel_std),
                          self.vehicles[vehicle]["absolute_position"] + normal(0, self.observation_pos_std)]
                         for vehicle in self.sorted_ids]).T

    def render(self):
        print('current state/velocity:', self.state)

    def step(self, rl_actions):
        """
        See parent class
        Prior to performing base_env's step function, vehicles are allowed to enter the network
        if requested, and the lists of vehicle id's are updated.
        """
        # store the previous edge for each vehicle
        for veh_id in self.ids:
            self.current_edge[veh_id] = self.vehicles[veh_id]["edge"]

        # in addition to regular actions, vehicles in the braess paradox scenario can
        # can also choose routes
        self.choose_routes(self.controlled_ids)

        # continue with performing requested actions and updating the observation space
        output = super().step(rl_actions)

        for veh_id in self.ids:
            # update previous edge data
            self.prev_edge[veh_id] = self.current_edge[veh_id]

            current_edge = self.vehicles[veh_id]["edge"]

            # update the vehicle entrance and exit times
            if current_edge in ["AC", "AD"] and self.prev_edge[veh_id] not in ["AC", "AD"]:
                self.current_route_times[veh_id]["enter"] = self.timer
                self.current_braess_route_choice[veh_id] = [self.vehicles[veh_id]["edge"]]

            elif current_edge == "CD" and self.prev_edge[veh_id] != "CD":
                self.current_braess_route_choice[veh_id].append(self.vehicles[veh_id]["edge"])

            elif current_edge in ["CB", "DB"] and self.prev_edge[veh_id] not in ["CB", "DB"]:
                self.current_braess_route_choice[veh_id].append(self.vehicles[veh_id]["edge"])
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
            self.prev_edge[veh_id] = self.vehicles[veh_id]["edge"]
            self.current_edge[veh_id] = self.vehicles[veh_id]["edge"]

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
            this_edge = self.vehicles[veh_id]["edge"]
            this_pos = self.vehicles[veh_id]["position"]
            edge_len = self.scenario.edge_len
            curve_len = self.scenario.curve_len

            # this term determines if the vehicle in question is close enough to any route-choosing nodes
            near_route_choosing_node = ((this_pos >= edge_len - 10) and (this_edge in ["DB", "CB"])) \
                or ((this_pos >= curve_len - 10) and (this_edge == "BA2"))

            # if the vehicle is not close enough, do not make any route changing decisions
            if not near_route_choosing_node:
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
                    current_route_choice_indx = random.choice([0, 2])
                else:
                    # the vehicle chooses the route that will allow it to move fastest
                    # choices between equivalent routes are made randomly
                    # current_route_choice_indx = random.choice(np.where(min(tt) == tt)[0])
                    current_route_choice_indx = np.argmin(tt)

                route_choices[i] = self.available_route_choices[current_route_choice_indx]

            self.traci_connection.vehicle.setRoute(vehID=veh_id, edgeList=route_choices[i])

    def apply_acceleration(self, veh_ids, acc):
        """
        See parent class
        """
        i_called = []  # index of vehicles in the list that have already received traci calls
        for i, veh_id in enumerate(veh_ids):
            this_edge = self.vehicles[veh_id]["edge"]
            this_pos = self.vehicles[veh_id]["position"]
            edge_len = self.scenario.edge_len

            # vehicles on edges CD and any of the edges not part of the braess paradox network
            # move very fast (mimicking a zero or negligible travel time)
            if this_edge in ["B", "BA1", "BA2"]:
                target_speed = self.scenario.net_params["AD_CB_speed_limit"]
                self.traci_connection.vehicle.slowDown(vehID=veh_id, speed=target_speed, duration=1)
                i_called.append(i)

            # elif this_edge in ["AC", "D"] and (this_pos > edge_len - 10 or this_pos < 10):
            #     target_speed = self.scenario.net_params["AC_DB_speed_limit"]  # speed limit on these lanes
            #     self.traci_connection.vehicle.slowDown(vehID=veh_id, speed=target_speed, duration=1)
            #     i_called.append(i)
            #
            # elif this_edge in ["AD", "CB"] and (this_pos > edge_len - 10 or this_pos < 10):
            #     target_speed = self.scenario.net_params["AD_CB_speed_limit"]  # speed limit on these lanes
            #     self.traci_connection.vehicle.slowDown(vehID=veh_id, speed=target_speed, duration=1)
            #     i_called.append(i)

        # delete data on vehicles that have already received traci calls in order to reduce run-time
        veh_ids = [veh_ids[i] for i in range(len(veh_ids)) if i not in i_called]
        acc = [acc[i] for i in range(len(acc)) if i not in i_called]

        # print([veh_id for veh_id in self.ids])
        # print([self.vehicles[veh_id]["leader"] for veh_id in self.ids])
        # print([self.vehicles[veh_id]["speed"] for veh_id in self.ids])
        # print([self.vehicles[veh_id]["headway"] for veh_id in self.ids])
        # print("---------------------------------------")

        super().apply_acceleration(veh_ids=veh_ids, acc=acc)

    def additional_command(self):
        """
        See parent class
        In order to mimic variable speed limits, the desired speed limit on the IDMController (v0) is modified
        depending on the lane the vehicle is located.
        """
        for veh_id in self.controlled_ids:
            current_edge = self.vehicles[veh_id]["edge"]
            # previous_edge = self.vehicles[veh_id]["previous_edge"]

            if current_edge in ["AD", "CB"] and self.prev_edge[veh_id] not in ["AD", "CB"]:
                self.vehicles[veh_id]["controller"].v0 = self.scenario.net_params["AD_CB_speed_limit"]

            elif current_edge in ["AC", "DB"] and self.prev_edge[veh_id] not in ["AC", "DB"]:
                self.vehicles[veh_id]["controller"].v0 = self.scenario.net_params["AC_DB_speed_limit"]

    # def apply_acceleration(self, veh_ids, acc=None, **kwargs):
    #     """
    #     See parent class
    #     """
    #     for i, veh_id in enumerate(veh_ids):
    #         target_speed = None
    #
    #         # vehicles on edges AC and DB move at a speed dependent on the density
    #         if self.vehicles[veh_id]["edge"] in ["AC", "D"] and \
    #                 (self.prev_edge[veh_id] not in ["AC", "D"] or veh_id in self.rl_ids):
    #             # compute the density of cars in the edge the vehicle is currently located
    #             this_edge = self.vehicles[veh_id]["edge"]
    #             num_cars = sum([self.vehicles[vID]["edge"] == this_edge for vID in self.ids])
    #             density = num_cars / self.scenario.edge_len
    #
    #             target_speed = self.varying_edge_speed(density)
    #
    #         # vehicles on edges AD and CB moves at a speed independent of the density
    #         elif self.vehicles[veh_id]["edge"] in ["AD", "CB"] and \
    #                 (self.prev_edge[veh_id] not in ["AD", "CB"] or veh_id in self.rl_ids):
    #             target_speed = self.constant_edge_speed
    #
    #         # vehicles on edges CD and any of the edges not part of the braess paradox network
    #         # move very fast (mimicking a zero or negligible travel time)
    #         elif self.vehicles[veh_id]["edge"] in ["CD", "B", "BA1", "BA2"] and \
    #                 (self.prev_edge[veh_id] not in ["CD", "B", "BA1", "BA2"] or veh_id in self.rl_ids):
    #             target_speed = 30.  # something fast
    #
    #         # rl vehicles can move in a fraction of the target velocity (to slow down the network)
    #         if veh_id in self.rl_ids and target_speed is not None:
    #             target_speed = target_speed * kwargs["speed_fraction"][i]
    #             self.traci_connection.vehicle.slowDown(veh_id, target_speed, 1)
    #
    #         elif target_speed is not None:
    #             self.traci_connection.vehicle.setSpeed(veh_id, target_speed)
