from cistar.envs.loop import LoopEnvironment

from rllab.spaces import Box
from rllab.spaces import Product
from rllab.spaces.discrete import Discrete

import traci
import pdb
import numpy as np


class SimpleLaneChangingAccelerationEnvironment(LoopEnvironment):
    """
    Fully functional environment. Takes in an *acceleration* as an action. Reward function is negative norm of the
    difference between the velocities of each vehicle, and the target velocity. State function is a vector of the
    velocities for each vehicle.
    """

    def __init__(self, env_params, sumo_binary, sumo_params, scenario):
        """ Initializes a lane-changing environment.
        Appends to the simple acceleration environment

        :param env_params:
        :param sumo_binary:
        :param sumo_params:
        :param scenario:
        """
        super().__init__(env_params, sumo_binary, sumo_params, scenario)

        if "lane_change_duration" in self.env_params:
            self.lane_change_duration = self.env_params['lane_change_duration'] / self.time_step
        else:
            self.lane_change_duration = 5 / self.time_step

        print(self.lane_change_fail_safe)
        print(self.fail_safe)

    @property
    def action_space(self):
        """
        Actions are:
         - a (continuous) acceleration from max-deacc to max-acc
         - a (discrete) direction with 3 values: 0) no lane change, 1) lane change to index +1,
                                                 2) lane change to index -1
        :return:
        """
        # acc_space = Box(low=-abs(self.env_params["max-deacc"]),
        #                 high=self.env_params["max-acc"],
        #                 shape=(self.scenario.num_rl_vehicles,))
        #
        # direction_space = Product(*[Discrete(3) for _ in range(self.scenario.num_rl_vehicles)])
        #
        # return Product([acc_space, direction_space])

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
        lane = Box(low=0, high=self.scenario.lanes-1, shape=(self.scenario.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        return Product([speed, lane, absolute_pos])

    def compute_reward(self, state, action, fail=False):
        """
        See parent class
        """
        if any(state[0] < 0) or fail:
            return -20.0

        max_cost = np.array([self.env_params["target_velocity"]]*self.scenario.num_vehicles)
        max_cost = np.linalg.norm(max_cost)

        cost = state[0] - self.env_params["target_velocity"]
        cost = np.linalg.norm(cost)

        return max_cost - cost

    def getState(self):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: an array of vehicle speed for each vehicle
        """
        return np.array([[self.vehicles[vehicle]["speed"],
                          self.vehicles[vehicle]["lane"],
                          self.vehicles[vehicle]["absolute_position"]] for vehicle in self.vehicles]).T

    def render(self):
        print('current velocity, lane, headway:', self.state)

    def apply_rl_actions(self, actions):
        """
        Takes a tuple and applies a lane change or acceleration. if a lane change is applied,
        don't issue any commands for the duration of the lane change and return negative rewards
        for actions during that lane change. if a lane change isn't applied, and sufficient time
        has passed, issue an acceleration like normal
        :param actions: (acceleration, lc_value, direction)
        :return: array of resulting actions: 0 if successful + other actions are ok, -1 if unsucessful / bad actions.
        """
        resulting_behaviors = []

        for i, veh_id in enumerate(self.rl_ids):
            # TODO: in the discrete scheme, we should get rid of lc_value and have direction be {-1,0,1}
            acceleration = actions[3 * i]
            lc_value = actions[3 * i + 1]
            direction = actions[3 * i + 2]

            # if self.timer > self.lane_change_duration + self.vehicles[veh_id]['last_lc']:
            #     # if enough time has elapsed since the last lane change, perform lane changes as requested
            #     resulting_behaviors.append(0)
            # else:
            #     # if not enough time has passed, ignore the acceleration action (set to zero)
            #     acceleration = 0.
            #
            #     # TODO: this is always going to be continuous, so we can't possibly punish them for having it non-zero
            #     # TODO: if we specify a range of accelerations that are acceptable during a lane change, then we can
            #     # TODO: start punishing
            #     if actions[3 * i] != 0:
            #         # if requested acceleration was non-zero, add a penalty to the reward fn
            #         resulting_behaviors.append(-1)
            #         # TODO: update the action to be the same as actual action?
            #         actions[3 * i] = 0.
            #     else:
            #         resulting_behaviors.append(0)

            # TODO: we need fail-safes to be outside of lane-changing control to ensure cars don't crash,
            # TODO: but should we penalize lane-changing cars for applying a fail-safe?
            self.apply_accel(veh_id, acc=acceleration)

            if lc_value > 0:
                # desired lc
                if self.timer > self.lane_change_duration + self.vehicles[veh_id]['last_lc']:
                    # enough time has passed, change lanes
                    lc_reward = self.apply_lane_change(veh_id, direction=direction)
                    resulting_behaviors.append(lc_reward)
                else:
                    # rl vehicle desires lane change but duration of previous lane change has not yet completed
                    resulting_behaviors.append(-1)
            else:
                resulting_behaviors.append(0)

        return resulting_behaviors

    def reset(self):
        observation = super().reset()

        for veh_id in self.rl_ids:
            self.vehicles[veh_id]['last_lc'] = -1 * self.lane_change_duration
        return observation


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

    def compute_reward(self, state, action, fail=False):
        """
        See parent class
        """
        if any(state[0] < 0) or fail:
            return -20.0

        # upper bound used to ensure the reward is always positive
        if np.any(state < 0):
            return -20.0
        max_cost = np.append(np.array([self.env_params["target_velocity_aggressive"]]*len(self.ind_nonaggressive)),
                             np.array([self.env_params["target_velocity"]]*len(self.ind_nonaggressive)))
        max_cost = np.linalg.norm(max_cost)

        # cost associated with being away from target velocity
        # if the vehicle's velocity is more than twice the target velocity, the cost does not become worse
        cost = np.append(state[0][self.ind_aggressive].clip(max=2*self.env_params["target_velocity_aggressive"]) -
                         self.env_params["target_velocity_aggressive"],
                         state[0][self.ind_nonaggressive].clip(max=2*self.env_params["target_velocity"]) -
                         self.env_params["target_velocity"])
        cost = np.linalg.norm(cost)

        return max_cost - cost
