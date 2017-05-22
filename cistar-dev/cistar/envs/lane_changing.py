from cistar.envs.loop import LoopEnvironment

from rllab.spaces import Box
from rllab.spaces import Product
import traci
import pdb
import numpy as np


"""
Fully functional environment. Takes in an *acceleration* as an action. Reward function is negative norm of the
difference between the velocities of each vehicle, and the target velocity. State function is a vector of the
velocities for each vehicle.
"""
class SimpleLaneChangingAccelerationEnvironment(LoopEnvironment):

    def __init__(self, env_params, sumo_binary, sumo_params, scenario):
        super().__init__(env_params, sumo_binary, sumo_params, scenario)

        if "lane_change_duration" in self.env_params:
            self.lane_change_duration = self.env_params['lane_change_duration'] / self.time_step
        else:
            self.lane_change_duration = 5 / self.time_step

        for rl_id in self.rl_ids:
            self.vehicles[rl_id]['last_lc'] = -1 * self.lane_change_duration

    @property
    def action_space(self):
        """
        Actions are a set of accelerations from 0 to 15m/s
        :return:
        """
        # TODO: max and min are parameters
        # accelerations = Box(low=, high=self.env_params["max-acc"], shape=(self.scenario.num_rl_vehicles, ))
        # lc_threshold = Box(low=-1, high=1, shape=(self.scenario.num_rl_vehicles, ))
        # left_right_threshold = Box(low=-1, high=1, shape=(self.scenario.num_rl_vehicles, ))

        lb = [-abs(self.env_params["max-deacc"]), -1, -1] * self.scenario.num_rl_vehicles
        ub = [self.env_params["max-acc"], 1, 1] * self.scenario.num_rl_vehicles
        return Box(np.array(lb), np.array(ub))

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        speed = Box(low=-np.inf, high=np.inf, shape=(self.scenario.num_vehicles,))
        lane = Box(low=0, high=self.scenario.lanes, shape=(self.scenario.num_vehicles,))
        headway = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        return Product([speed, lane, headway])

    def apply_action(self, veh_id, action):
        """
        See parent class (base_env)
         Given an acceleration, set instantaneous velocity given that acceleration.
        """
        thisSpeed = self.vehicles[veh_id]['speed']
        nextVel = thisSpeed + action * self.time_step
        nextVel = max(0, nextVel)
        # if we're being completely mathematically correct, 1 should be replaced by int(self.time_step * 1000)
        # but it shouldn't matter too much, because 1 is always going to be less than int(self.time_step * 1000)
        self.traci_connection.vehicle.slowDown(veh_id, nextVel, 1)

    def compute_reward(self, state, action):
        """
        See parent class
        """
        if any(state[0] < 0):
            return -20.0
        max_cost = np.array([self.env_params["target_velocity"]]*self.scenario.num_vehicles)
        max_cost = np.linalg.norm(max_cost)

        cost = state[0] - self.env_params["target_velocity"]
        cost = np.linalg.norm(cost)
        return max_cost - cost
        #return -np.linalg.norm(velocity - self.env_params["target_velocity"])

    def getState(self):
        """
       See parent class
       The state is an array the velocities for each vehicle
       :return: an array of vehicle speed for each vehicle
       """
        return np.array([[self.vehicles[vehicle]["speed"],
                          self.vehicles[vehicle]["lane"],  # TODO: what if we have more than 10 lanes?
                          self.get_headway(vehicle)]for vehicle in self.vehicles]).T

    def render(self):
        print('current velocity, lane, headway:', self.state)

    def change_lanes(self, veh_id, direction):
        """
        Makes the traci call to change an rl-vehicle's lane based on a direction
        :param veh_id: vehicle to apply the lane change to
        :param direction: double between -1 and 1, -1 to the left 1 to the right
        :return: (1, timer) for successful lane change, 0 for unsuccessful but possible lane changing
        (i.e. may be vaild, but unsafe), -1 for impossible lane change
        """

        if self.scenario.lanes == 1:
            print("Uh oh, single lane track.")
            return -1
        else:
            curr_lane = self.vehicles[veh_id]['lane']
            if direction > 0:
                # lane change right
                # print("right")
                if curr_lane > 0:
                    self.traci_connection.vehicle.changeLane(veh_id, int(curr_lane-1), int(self.lane_change_duration)) # might be flipped??
                    self.vehicles[veh_id]['last_lc'] = self.timer
                    return 1
                else:
                    return -1
            else:
                # lane change left
                # print("left")
                if curr_lane < self.scenario.lanes - 1:
                    self.traci_connection.vehicle.changeLane(veh_id, int(curr_lane+1), int(self.lane_change_duration))  # might be flipped??
                    self.vehicles[veh_id]['last_lc'] = self.timer
                    return 1
                else:
                    return -1

    def apply_rl_actions(self, actions):
        """
        Takes a tuple and applies a lane change or acceleration. if a lane change is applied,
        don't issue any commands for the duration of the lane change and return negative rewards
        for actions during that lane change. if a lane change isn't applied, and sufficient time
        has passed, issue an acceleration like normal
        :param actions: (acceleraton, lc_value, direction)
        :return: array of resulting actions: 1 if successful + other actions are ok, -1 if unsucessful / bad actions.
        """

        resulting_behaviors = []

        for i, veh_id in enumerate(self.rl_ids):
            # if veh_id == "rl_1":
            #     print(actions[3*i], actions[3*i+1], actions[3*i+2])
            lc_value = actions[3 * i + 1]
            direction = actions[3 * i + 2]

            acceleration = actions[3 * i]
            if self.fail_safe == 'instantaneous':
                safe_action = self.vehicles[veh_id]['controller'].get_safe_action_instantaneous(self, acceleration)
            elif self.fail_safe == 'eugene':
                safe_action = self.vehicles[veh_id]['controller'].get_safe_action(self, acceleration)
            else:
                safe_action = acceleration
            self.apply_action(veh_id, action=safe_action)

            successful_lc = 0

            if lc_value > 0:
                # desired lc
                if self.timer > self.lane_change_duration + self.vehicles[veh_id]['last_lc']:
                    # enough time has passed, change lanes
                    successful_lc = self.change_lanes(veh_id, direction)
                else:
                    # desired lane change but duration has not completed
                    resulting_behaviors.append(-1)  # something negative to add to reward fn

            if successful_lc != 1:
                resulting_behaviors.append(1)  # something positive to add to reward fn
            elif successful_lc == 1:
                # changed lanes
                resulting_behaviors.append(
                    -1)  # something negative to add to reward fn if desired acceleration is large

        return resulting_behaviors


class ShepherdAggressiveDrivers(SimpleLaneChangingAccelerationEnvironment):

    def __init__(self, env_params, sumo_binary, sumo_params, scenario):
        super().__init__(env_params, sumo_binary, sumo_params, scenario)

        # index of aggressive vehicles
        self.ind_aggressive = env_params["ind_aggressive"]

        # index of non-aggressive vehicles
        ind_nonaggressive = np.arange(self.scenario.num_vehicles)
        ind_nonaggressive = ind_nonaggressive[np.array([ind_nonaggressive[i] not in self.ind_aggressive
                                                        for i in range(len(ind_nonaggressive))])]
        self.ind_nonaggressive = ind_nonaggressive

    def compute_reward(self, velocity, action):
        """
        See parent class
        """
        # upper bound used to ensure the reward is always positive
        max_cost = np.append(np.array([self.env_params["target_velocity_aggressive"]]*len(self.ind_nonaggressive)),
                             np.array([self.env_params["target_velocity"]]*len(self.ind_nonaggressive)))
        max_cost = np.linalg.norm(max_cost)

        # cost associated with being away from target velocity
        # if the vehicle's velocity is more than twice the target velocity, the cost does not become worse
        cost = np.append(velocity[self.ind_aggressive].clip(max=2*self.env_params["target_velocity_aggressive"]) -
                         self.env_params["target_velocity_aggressive"],
                         velocity[self.ind_nonaggressive].clip(max=2*self.env_params["target_velocity"]) -
                         self.env_params["target_velocity"])
        cost = np.linalg.norm(cost)

        #######################################
        if any(velocity < 0):
            print(velocity)
        #######################################

        return max_cost - cost
