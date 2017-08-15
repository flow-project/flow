from cistar_dev.envs.loop import LoopEnvironment
from cistar_dev.core import multi_agent_rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import traci

import numpy as np
from numpy.random import normal

import pdb


class TwoIntersectionMultiAgentEnvironment(LoopEnvironment):
    """
    Fully functional environment. Takes in an *acceleration* as an action. Reward function is negative norm of the
    difference between the velocities of each vehicle, and the target velocity. State function is a vector of the
    velocities for each vehicle.
    """

    @property
    def action_space(self):
        """
        Actions are a set of accelerations from 0 to 15m/s
        :return:
        """
        action_space = []
        for veh_id in self.rl_ids:
            action_space.append(Box(low=self.env_params["max-deacc"], 
                high=self.env_params["max-acc"], shape=(1, )))
        return action_space

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        observation_space = []
        speed = Box(low=0, high=np.inf, shape=(self.scenario.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))
        #dist_to_intersection = Box(low=-np.inf, high=np.inf, shape=(self.scenario.num_vehicles,))
        obs_tuple = Tuple((speed, absolute_pos))
        for veh_id in self.rl_ids:
            observation_space.append(obs_tuple)
        return observation_space

    def apply_rl_actions(self, rl_actions):
        """
        See parent class
        """
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]
        for i, veh_id in enumerate(sorted_rl_ids):
            # If we are outside the control region, just get up to speed
            if (self.get_distance_to_intersection(veh_id)[0] > 150 and 
                    (self.vehicles[veh_id]["edge"] == "bottom" or 
                    self.vehicles[veh_id]["edge"] == "left")):
                # get up to max speed
                if self.vehicles[veh_id]["speed"] < self.scenario.initial_config["enter_speed"]:
                    # accelerate as fast as you can
                    if ((self.scenario.initial_config["enter_speed"] - 
                        self.vehicles[veh_id]["speed"])/self.time_step > self.env_params["max-acc"]):
                        rl_actions[i][0][0] =  self.env_params["max-acc"]
                    # accelerate up to target velocity
                    else:
                        rl_actions[i][0][0] = ((self.scenario.initial_config["enter_speed"] - 
                                            self.vehicles[veh_id]["speed"])/self.time_step)
                # at max speed, don't accelerate
                else:
                    rl_actions[i][0][0] = 0.0
            # we are at the intersection, turn the fail-safe on
            # make it so this only happens once
            else:
                self.traci_connection.vehicle.setSpeedMode(veh_id, 1)
            if (self.vehicles[veh_id]["speed"] + 
                self.time_step*rl_actions[i][0] > self.vehicles[veh_id]["max_speed"]):
                rl_actions[i][0][0] = ((self.vehicles[veh_id]["max_speed"] - 
                                            self.vehicles[veh_id]["speed"])/self.time_step)

        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        # 
        return multi_agent_rewards.desired_velocity(
            state, rl_actions, fail=kwargs["fail"], target_velocity=self.env_params["target_velocity"])
        # return rewards.min_delay(state, rl_actions, target_velocity=self.env_params["target_velocity"],
        #     time_step=self.sumo_params["time_step"], fail=kwargs["fail"])

    def getState(self, **kwargs):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        # if kwargs["observability"] == "full":
        # full observability
        # return np.array([[self.vehicles[veh_id]["speed"] + normal(0, self.observation_vel_std),
        #                   self.vehicles[veh_id]["absolute_position"] + normal(0, self.observation_pos_std)]
        #                  for veh_id in self.sorted_ids]).T

        # return np.array([[self.vehicles[veh_id]["speed"],
        #                   self.vehicles[veh_id]["absolute_position"],
        #                   self.get_distance_to_intersection(veh_id)[0]]
        #                  for veh_id in self.sorted_ids]).T

        obs_arr = []
        for i in range(self.scenario.num_vehicles):
            speed = [self.vehicles[veh_id]["speed"] for veh_id in self.sorted_ids]
            abs_pos = [self.vehicles[veh_id]["absolute_position"] for veh_id in self.sorted_ids]
            tup = (speed, abs_pos)
            obs_arr.append(tup)
        return obs_arr
        # else:
        #     # partial observability (n car ahead, m car behind)
        #     sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]
        #     veh_ids = []
        #     for veh_id in sorted_rl_ids:
        #         veh_ids.append(veh_id)  # add rl vehicle
        #         veh_ids.append(self.vehicles[veh_id]["leader"])  # add vehicle in front of rl vehicle
        #         veh_ids.append(self.vehicles[veh_id]["follower"])  # add vehicle behind rl vehicle
        #
        #     veh_ids = np.unique(veh_ids)  # remove redundant vehicle ids



        # partial observability (2 cars ahead, 2 cars behind)

    # def _render(self):
    #     print('current state/velocity:', self.state)
