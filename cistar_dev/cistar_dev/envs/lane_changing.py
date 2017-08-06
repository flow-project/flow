from cistar_dev.envs.loop import LoopEnvironment
from cistar_dev.core import rewards
from cistar_dev.controllers.car_following_models import IDMController

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
from gym.spaces.discrete import Discrete

import traci
import pdb
import numpy as np
from numpy.random import normal
import time


class SimpleLaneChangingAccelerationEnvironment(LoopEnvironment):
    """
    Fully functional environment. Takes in an *acceleration* as an action. Reward function is negative norm of the
    difference between the velocities of each vehicle, and the target velocity. State function is a vector of the
    velocities for each vehicle.
    """

    @property
    def action_space(self):
        """
        Actions are:
         - a (continuous) acceleration from max-deacc to max-acc
         - a (discrete) direction with 3 values: 0) lane change to index -1, 1) no lane change,
                                                 2) lane change to index +1
        :return:
        """
        # action_space = Product(*[Discrete(3) for _ in range(self.scenario.num_rl_vehicles)],
        #     Box(low=-abs(self.env_params["max-deacc"]),
        #                 high=self.env_params["max-acc"],
        #                 shape=(self.scenario.num_rl_vehicles,)))
        #
        # return action_space

        lb = [-abs(self.env_params["max-deacc"]), -1] * self.scenario.num_rl_vehicles
        ub = [self.env_params["max-acc"], 1] * self.scenario.num_rl_vehicles
        return Box(np.array(lb), np.array(ub))

    @property
    def observation_space(self):
        """
        See parent class
        An observation consists of the velocity, lane index, and absolute position of each vehicle
        in the fleet
        """

        speed = Box(low=-np.inf, high=np.inf, shape=(self.scenario.num_vehicles,))
        lane = Box(low=0, high=self.scenario.lanes-1, shape=(self.scenario.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        return Tuple((speed, lane, absolute_pos))

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        target_velocity = self.env_params["target_velocity"]

        # compute the system-level performance of vehicles from a velocity perspective
        reward = rewards.desired_velocity(state, rl_actions, fail=kwargs["fail"], target_velocity=target_velocity)

        # punish excessive lane changes by reducing the reward by a set value every time an rl car changes lanes
        for veh_id in self.rl_ids:
            if self.vehicles[veh_id]["last_lc"] == self.timer:
                reward -= 1

        return reward

    def getState(self):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: an array of vehicle speed for each vehicle
        """
        # return np.array([[self.vehicles[veh_id]["speed"] + normal(0, self.observation_vel_std),
        #                   self.vehicles[veh_id]["absolute_position"] + normal(0, self.observation_pos_std),
        #                   self.vehicles[veh_id]["lane"]] for veh_id in self.sorted_ids]).T

        # for moving bottleneck: use only local trailing data for the moving bottleneck experiment
        vehID = self.rl_ids[0]

        trail_lane0 = self.get_trailing_car(vehID, lane=0)
        headway_lane0 = (self.vehicles[vehID]["absolute_position"] - self.vehicles[trail_lane0]["absolute_position"]) \
            % self.scenario.length

        trail_lane1 = self.get_trailing_car(vehID, lane=1)
        headway_lane1 = (self.vehicles[vehID]["absolute_position"] - self.vehicles[trail_lane1]["absolute_position"]) \
            % self.scenario.length

        if self.scenario.lanes == 2:
            return np.array([[self.vehicles[vehID]["speed"], self.vehicles[trail_lane0]["speed"],
                              self.vehicles[trail_lane1]["speed"]],
                             [self.vehicles[vehID]["absolute_position"], headway_lane0, headway_lane1],
                             [self.vehicles[vehID]["lane"], self.vehicles[trail_lane0]["lane"],
                              self.vehicles[trail_lane1]["lane"]]])

        elif self.scenario.lanes == 3:
            trail_lane2 = self.get_trailing_car(vehID, lane=2)
            headway_lane2 = \
                (self.vehicles[vehID]["absolute_position"] - self.vehicles[trail_lane2]["absolute_position"]) \
                % self.scenario.length

            return np.array([[self.vehicles[vehID]["speed"], self.vehicles[trail_lane0]["speed"],
                              self.vehicles[trail_lane1]["speed"], self.vehicles[trail_lane2]["speed"]],
                             [self.vehicles[vehID]["absolute_position"], headway_lane0, headway_lane1, headway_lane2],
                             [self.vehicles[vehID]["lane"], self.vehicles[trail_lane0]["lane"],
                              self.vehicles[trail_lane1]["lane"], self.vehicles[trail_lane2]["lane"]]])


    # def render(self):
    #     print('current velocity, lane, absolute_pos, headway:', self.state)

    def apply_rl_actions(self, actions):
        """
        Takes a tuple and applies a lane change or acceleration. if a lane change is applied,
        don't issue any commands for the duration of the lane change and return negative rewards
        for actions during that lane change. if a lane change isn't applied, and sufficient time
        has passed, issue an acceleration like normal
        :param actions: (acceleration, lc_value, direction)
        :return: array of resulting actions: 0 if successful + other actions are ok, -1 if unsucessful / bad actions.
        """
        # acceleration = actions[-1]
        # direction = np.array(actions[:-1]) - 1

        acceleration = actions[::2]
        direction = np.round(actions[1::2])

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = [self.timer <= self.lane_change_duration + self.vehicles[veh_id]['last_lc']
                                 for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.apply_lane_change(sorted_rl_ids, direction=direction)








































class LaneChangeOnlyEnvironment(SimpleLaneChangingAccelerationEnvironment):

    def __init__(self, env_params, sumo_binary, sumo_params, scenario):
        super().__init__(env_params, sumo_binary, sumo_params, scenario)
        vehicle_controllers = {}
        for veh_id in self.rl_ids:
            vehicle_controllers[veh_id] = IDMController(veh_id)
        self.vehicle_controllers = vehicle_controllers

    @property
    def action_space(self):
        """
        Actions are:
         - a (discrete) direction with 3 values: 0) lane change to index -1, 1) no lane change,
                                                 2) lane change to index +1
        :return:
        """

        lb = [-1] * self.scenario.num_rl_vehicles
        ub = [1] * self.scenario.num_rl_vehicles
        return Box(np.array(lb), np.array(ub))

    def apply_rl_actions(self, actions):
        """
        Takes a tuple and applies a lane change. if a lane change is applied,
        don't issue another lane change for the duration of the lane change. 
        Accelerations are given by an idm model. 
        :param actions: (acceleration, lc_value, direction)
        :return: array of resulting actions: 0 if successful + other actions are ok, -1 if unsucessful / bad actions.
        """
        # acceleration = actions[-1]
        # direction = np.array(actions[:-1]) - 1


        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]

        # get the accelerations from an idm model
        acceleration = np.zeros((len(sorted_rl_ids)))
        for i, rl_id in enumerate(sorted_rl_ids):
            acceleration[i] = self.vehicle_controllers[rl_id].get_action(self)

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)

        # get the lane changes from the neural net
        direction = np.round(actions)

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = [self.timer <= self.lane_change_duration + self.vehicles[veh_id]['last_lc']
                                 for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

        self.apply_lane_change(sorted_rl_ids, direction=direction)


        resulting_behaviors = []

        return resulting_behaviors


class RLOnlyLane(SimpleLaneChangingAccelerationEnvironment):

    def compute_reward(self, state, action, **kwargs):
        """
        See parent class
        """

        if any(state[0] < 0) or kwargs["fail"]:
            return -20.0

        #
        # flag = 1
        # # max_cost3 = np.array([self.env_params["target_velocity"]]*len(self.rl_ids))
        # # max_cost3 = np.linalg.norm(max_cost3)
        # # cost3 = [self.vehicles[veh_id]["speed"] - self.env_params["target_velocity"] for veh_id in self.rl_ids]
        # # cost3 = np.linalg.norm(cost)
        # # for i, veh_id in enumerate(self.rl_ids):
        # #     if self.vehicles[veh_id]["lane"] != 0:
        # #         flag = 1
        #
        # if flag:
        #     return max_cost - cost - cost2
        # else:
        #     return (max_cost - cost) + (max_cost3 - cost3) - cost2

        reward_type = 1

        if reward_type == 1:
            # this reward type only rewards the velocity of the rl vehicle if it is in lane zero
            # otherwise, the reward function perceives the velocity of the rl vehicles as 0 m/s

            max_cost = np.array([self.env_params["target_velocity"]]*self.scenario.num_vehicles)
            max_cost = np.linalg.norm(max_cost)

            vel = state[0]
            lane = state[1]
            vel[lane != 0] = np.array([0] * sum(lane != 0))

            cost = vel - self.env_params["target_velocity"]
            cost = np.linalg.norm(cost)

            return max(max_cost - cost, 0)

        elif reward_type == 2:
            # this reward type only rewards non-rl vehicles, and penalizes rl vehicles for being
            # in the wrong lane

            # reward for only non-rl vehicles
            max_cost = np.array([self.env_params["target_velocity"]]*len(self.controlled_ids))
            max_cost = np.linalg.norm(max_cost)

            cost = [self.vehicles[veh_id]["speed"] - self.env_params["target_velocity"]
                    for veh_id in self.controlled_ids]
            cost = np.linalg.norm(cost)

            # penalty for being in the other lane
            # calculate how long the cars have been in the left lane
            left_lane_cost = np.zeros(len(self.rl_ids))
            for i, veh_id in enumerate(self.rl_ids):
                if self.vehicles[veh_id]["lane"] != 0:
                    # method 1:
                    # if its possible to lane change and we are still hanging out in the left lane
                    # start penalizing it
                    # left_lane_cost[i] = np.max([0, (self.timer - self.vehicles[veh_id]['last_lc'] -
                    #                                 self.lane_change_duration)])

                    # method 2:
                    # penalize the left lane in increasing amount from the start
                    left_lane_cost[i] = self.timer/20

            cost2 = np.linalg.norm(np.array(left_lane_cost))/10

            return max_cost - cost - cost2

    @property
    def observation_space(self):
        """
        See parent class
        An observation consists of the velocity, lane index, and absolute position of each vehicle
        in the fleet
        """
        speed = Box(low=0, high=np.inf, shape=(self.scenario.num_vehicles,))
        lane = Box(low=0, high=self.scenario.lanes-1, shape=(self.scenario.num_vehicles,))
        pos = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        return Tuple((speed, lane, pos))

    def getState(self):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: an array of vehicle speed for each vehicle
        """
        # sorting states by position
        sorted_indx = np.argsort([self.vehicles[veh_id]["absolute_position"] for veh_id in self.ids])
        sorted_ids = np.array(self.ids)[sorted_indx]

        return np.array([[self.vehicles[veh_id]["speed"],
                          self.vehicles[veh_id]["lane"],
                          self.vehicles[veh_id]["absolute_position"]] for veh_id in sorted_ids])


    # def render(self):
    #     print('current velocity, lane, headway, adj headway:', self.state)


# class ShepherdAggressiveDrivers(SimpleLaneChangingAccelerationEnvironment):
#
#     def __init__(self, env_params, sumo_binary, sumo_params, scenario):
#         super().__init__(env_params, sumo_binary, sumo_params, scenario)
#
#         # index of aggressive vehicles
#         self.ind_aggressive = env_params["ind_aggressive"]
#
#         # index of non-aggressive vehicles
#         ind_nonaggressive = np.arange(self.scenario.num_vehicles)
#         ind_nonaggressive = ind_nonaggressive[np.array([ind_nonaggressive[i] not in self.ind_aggressive
#                                                         for i in range(len(ind_nonaggressive))])]
#         self.ind_nonaggressive = ind_nonaggressive
#
#     def compute_reward(self, state, action, **kwargs):
#         """
#         See parent class
#         """
#         # if any(state[0] < 0) or kwargs["fail"]:
#         #     return -20.0
#
#         # max_cost = np.append(np.array([self.env_params["target_velocity_aggressive"]]*len(self.ind_nonaggressive)),
#         #                      np.array([self.env_params["target_velocity"]]*len(self.ind_nonaggressive)))
#         # max_cost = np.linalg.norm(max_cost)
#
#         # # cost associated with being away from target velocity
#         # # if the vehicle's velocity is more than twice the target velocity, the cost does not become worse
#         # cost = np.append(state[0][self.ind_aggressive].clip(max=2*self.env_params["target_velocity_aggressive"]) -
#         #                  self.env_params["target_velocity_aggressive"],
#         #                  state[0][self.ind_nonaggressive].clip(max=2*self.env_params["target_velocity"]) -
#         #                  self.env_params["target_velocity"])
#         # cost = np.linalg.norm(cost)
#
#         # return max_cost - cost
#
#         if any(state[0] < 0) or kwargs["fail"]:
#             return -20.0
#
#         max_cost = np.append(np.array([self.env_params["target_velocity_aggressive"]]*len(self.ind_nonaggressive)),
#                              np.array([self.env_params["target_velocity"]]*len(self.ind_nonaggressive)))
#         max_cost = np.linalg.norm(max_cost)
#
#         # cost associated with being away from target velocity
#         # if the vehicle's velocity is more than twice the target velocity, the cost does not become worse
#         cost = np.append(state[0][self.ind_aggressive].clip(max=2*self.env_params["target_velocity_aggressive"]) -
#                          self.env_params["target_velocity_aggressive"],
#                          state[0][self.ind_nonaggressive].clip(max=2*self.env_params["target_velocity"]) -
#                          self.env_params["target_velocity"])
#         cost = np.linalg.norm(cost)
#
#         return max_cost - cost
