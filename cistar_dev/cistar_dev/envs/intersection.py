"""
This class is an extension the SumoEnvironment class located in base_env.

With this class, vehicles can enter the network following a probability model,
and exit as soon as they reach their destination.

(describe how the action and observation spaces are modified)
"""

from cistar_dev.core.base_env import SumoEnvironment
from cistar_dev.envs.loop import LoopEnvironment
from cistar_dev.controllers.base_controller import SumoController
from cistar_dev.controllers.rlcontroller import RLController

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np
from numpy.random import normal, uniform
from random import randint

import pdb


class SimpleIntersectionEnvironment(LoopEnvironment):
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
        self.vType = list(scenario.type_params.keys())
        self.prob_vType = np.array([scenario.type_params[key][0] for key in self.vType])
        self.prob_vType = self.prob_vType / sum(self.prob_vType)
        self.prob_vType = np.cumsum(self.prob_vType)

        # the entering speed of each car is set to the max speed of the lane the vehicle is entering
        self.enter_speed = dict()
        for key in scenario.net_params["speed_limit"]:
            self.enter_speed[key] = scenario.net_params["speed_limit"][key]

        # total vehicles contains the total number of vehicles of every type to enter the network.
        # this is used to provide new vehicles of the same type with appropriate vehicle ids
        self.total_vehicles = dict()
        for vehicle_type in self.vType:
            self.total_vehicles[vehicle_type] = 0

        super().__init__(env_params, sumo_binary, sumo_params, scenario)

    def step(self, rl_actions):
        """
        See parent class
        Prior to performing base_env's step function, vehicles are allowed to enter the network
        if requested, and the lists of vehicle id's are updated.
        """
        new_vehicles = False  # checks whether new vehicles were added in the current step
        new_ids = []          # contains the ids of all new vehicles added this step

        for entry in self.prob_enter.keys():
            # input variable to the probability function
            x = self.timer + 1 - self.last_enter_time[entry]

            # check if a vehicle wants to enter a lane
            if self.prob_enter[entry](x) > uniform(0, 1):
                new_vehicles = True
                self.last_enter_time[entry] = self.timer + 1

                # if a car wants to enter, determine which type it is
                vType_choice = uniform(0, 1)
                for i in range(len(self.prob_vType)):
                    if vType_choice <= self.prob_vType[i]:
                        new_type_id = self.vType[i]
                        self.total_vehicles[new_type_id] += 1
                        new_veh_id = self.vType[i] + '_' + str(self.total_vehicles[new_type_id])  #-1)
                        new_ids.append(new_veh_id)
                        break
                new_lane_index = randint(0, self.scenario.lanes[self.scenario.enter_lane[entry]]-1)
                new_route_id = "route" + self.scenario.enter_lane[entry]

                # add the car to the start of the lane (will not be available until the base_env step is performed)
                self.traci_connection.vehicle.add(new_veh_id, new_route_id, depart=0, pos=0,
                                                  speed=0,  # int(self.scenario.net_params["speed_limit"][entry]),
                                                  lane=new_lane_index, typeID=new_type_id)

        # continue with performing requested actions and updating the observation space
        super().step(rl_actions)

        if new_vehicles:
            for veh_id in new_ids:
                # add the initial conditions of the new vehicles to the self.vehicles dict
                self.vehicles[veh_id] = dict()
                self.vehicles[veh_id]["id"] = new_veh_id
                self.vehicles[veh_id]["type"] = self.traci_connection.vehicle.getTypeID(veh_id)
                self.vehicles[veh_id]["edge"] = self.traci_connection.vehicle.getRoadID(veh_id)
                self.vehicles[veh_id]["position"] = self.traci_connection.vehicle.getLanePosition(veh_id)
                self.vehicles[veh_id]["lane"] = self.traci_connection.vehicle.getLaneIndex(veh_id)
                self.vehicles[veh_id]["speed"] = self.traci_connection.vehicle.getSpeed(veh_id)
                self.vehicles[veh_id]["length"] = self.traci_connection.vehicle.getLength(veh_id)
                self.vehicles[veh_id]["max_speed"] = self.traci_connection.vehicle.getMaxSpeed(veh_id)
                self.vehicles[veh_id]["absolute_position"] = self.get_x_by_id(veh_id)
                self.vehicles[veh_id]['last_lc'] = -1 * self.lane_change_duration

                self.prev_last_lc[veh_id] = -1 * self.lane_change_duration

                self.set_speed_mode(veh_id)
                self.set_lane_change_mode(veh_id)

                controller_params = self.scenario.type_params[self.vehicles[veh_id]["type"]][1]
                self.vehicles[veh_id]['controller'] = controller_params[0](veh_id=veh_id, **controller_params[1])

                lane_changer_params = self.scenario.type_params[self.vehicles[veh_id]["type"]][2]
                if lane_changer_params is not None:
                    self.vehicles[veh_id]['lane_changer'] = \
                        lane_changer_params[0](veh_id=veh_id, **lane_changer_params[1])
                else:
                    self.vehicles[veh_id]['lane_changer'] = None

                # set the color of the new vehicle to comply with other vehicles in the network of the same type
                self.traci_connection.vehicle.setColor(new_veh_id, self.colors[new_type_id])

                # update the lists of vehicle ids
                self.ids.append(veh_id)
                if controller_params[0] == SumoController:
                    self.sumo_ids.append(veh_id)
                elif controller_params[0] == RLController:
                    self.rl_ids.append(veh_id)
                else:
                    self.controlled_ids.append(veh_id)

    @property
    def action_space(self):
        """
        See parent class
        """
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
            return Tuple((speed, absolute_pos))
        else:
            return Tuple((speed, lane, absolute_pos))

    def apply_rl_actions(self, rl_actions):
        """
        See parent class
        """
        # sorting states by position
        sorted_indx = np.argsort([self.vehicles[veh_id]["absolute_position"] for veh_id in self.rl_ids])
        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = np.array(self.rl_ids)[sorted_indx]

        if all([self.scenario.lanes[key] == 1 for key in self.scenario.lanes]):
            acceleration = rl_actions
            self.apply_acceleration(sorted_rl_ids, acc=acceleration)

        else:
            acceleration = rl_actions[::2]
            direction = np.round(rl_actions[1::2])

            # represents vehicles that are allowed to change lanes
            non_lane_changing_veh = [self.timer <= self.lane_change_duration + self.vehicles[veh_id]['last_lc']
                                     for veh_id in sorted_rl_ids]
            # vehicle that are not allowed to change have their directions set to 0
            direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

            self.apply_acceleration(sorted_rl_ids, acc=acceleration)
            self.apply_lane_change(sorted_rl_ids, direction=direction)

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
        sorted_indx = np.argsort([self.vehicles[veh_id]["absolute_position"] for veh_id in self.ids])
        sorted_ids = np.array(self.ids)[sorted_indx]

        if self.scenario.lanes == 1:
            return np.array([[self.vehicles[vehicle]["speed"] + normal(0, kwargs["observation_vel_std"]),
                              self.vehicles[vehicle]["absolute_position"] + normal(0, kwargs["observation_pos_std"])]
                             for vehicle in sorted_ids]).T
        else:
            return np.array([[self.vehicles[veh_id]["speed"] + normal(0, kwargs["observation_vel_std"]),
                              self.vehicles[veh_id]["absolute_position"] + normal(0, kwargs["observation_pos_std"]),
                              self.vehicles[veh_id]["lane"]] for veh_id in sorted_ids]).T

    # def render(self):
    #     print('current state/velocity:', self.state)

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
