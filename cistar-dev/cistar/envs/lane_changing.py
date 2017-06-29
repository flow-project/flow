from cistar.envs.loop import LoopEnvironment

from rllab.spaces import Box
from rllab.spaces import Product
from rllab.spaces.discrete import Discrete

import traci
import pdb
import numpy as np
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
        # is_rl = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        # return Product([speed, lane, absolute_pos, is_rl])
        return Product([speed, lane, absolute_pos])

    def compute_reward(self, state, action, **kwargs):
        """
        See parent class
        """
        if any(state[0] < 0) or kwargs["fail"]:
            return -20.0

        max_cost = np.array([self.env_params["target_velocity"]]*self.scenario.num_vehicles)
        max_cost = np.linalg.norm(max_cost)

        cost = state[0] - self.env_params["target_velocity"]
        cost = np.linalg.norm(cost)

        return max(max_cost - cost, 0)

    def getState(self):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: an array of vehicle speed for each vehicle
        """
        sort_pos = True

        if sort_pos:
            # sorting states by position
            sorted_indx = np.argsort([self.vehicles[veh_id]["absolute_position"] for veh_id in self.ids])

            sorted_ids = np.array(self.ids)[sorted_indx]

            sorted_rl_cars = np.array([0] * len(sorted_ids))
            sorted_rl_cars[:len(self.rl_ids)] = np.sort([self.vehicles[veh_id]["absolute_position"]
                                                         for veh_id in self.rl_ids])

            state = np.array([[self.vehicles[veh_id]["speed"],
                              self.vehicles[veh_id]["lane"],
                              self.vehicles[veh_id]["absolute_position"]] for veh_id in sorted_ids]).T
            # state = np.vstack((state, sorted_rl_cars))

            return state
        else:
            return np.array([[self.vehicles[veh_id]["speed"],
                              self.vehicles[veh_id]["lane"],
                              self.vehicles[veh_id]["absolute_position"]] for veh_id in self.ids]).T

    def render(self):
        print('current velocity, lane, absolute_pos, headway:', self.state)

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

        sort_pos = True
        
        if sort_pos:
            # sorting states by position
            sorted_indx = np.argsort([self.vehicles[veh_id]["absolute_position"] for veh_id in self.rl_ids])

            # re-arrange actions according to mapping in observation space
            sorted_rl_ids = np.array(self.rl_ids)[sorted_indx]

            # represents vehicles that are allowed to change lanes
            non_lane_changing_veh = [self.timer <= self.lane_change_duration + self.vehicles[veh_id]['last_lc']
                                     for veh_id in sorted_rl_ids]
            direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

            self.apply_acceleration(sorted_rl_ids, acc=acceleration)
            self.apply_lane_change(sorted_rl_ids, direction=direction)
        else:
            # represents vehicles that are allowed to change lanes
            non_lane_changing_veh = [self.timer <= self.lane_change_duration + self.vehicles[veh_id]['last_lc']
                                     for veh_id in self.rl_ids]
            direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

            self.apply_acceleration(self.rl_ids, acc=acceleration)
            self.apply_lane_change(self.rl_ids, direction=direction)

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
        return Product([speed, lane, pos])

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
                          self.vehicles[veh_id]["absolute_position"]] for veh_id in sorted_ids]).T


    def render(self):
        print('current velocity, lane, headway, adj headway:', self.state)


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
