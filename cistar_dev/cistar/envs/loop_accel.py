from cistar.envs.loop import LoopEnvironment
from cistar.core import rewards
from cistar.core import multi_agent_rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np
from numpy.random import normal


class SimpleAccelerationEnvironment(LoopEnvironment):
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
        return Box(low=-np.abs(self.env_params.get_additional_param("max-deacc")),
                   high=self.env_params.get_additional_param("max-acc"),
                   shape=(self.vehicles.num_rl_vehicles, ))

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        self.obs_var_labels = ["Velocity", "Absolute_pos"]
        speed = Box(low=0, high=np.inf, shape=(self.vehicles.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))
        return Tuple((speed, absolute_pos))

    def apply_rl_actions(self, rl_actions):
        """
        See parent class
        """
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]
        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        # reward desired velocity
        reward = rewards.desired_velocity(self, fail=kwargs["fail"])

        # # punish small headways
        # headway_penalty = rewards.punish_small_rl_headways(
        #     self.vehicles, rl_ids=self.rl_ids, headway_threshold=20, penalty_gain=0.75)
        #
        # reward += headway_penalty

        return reward

    def get_state(self, **kwargs):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        scaled_rel_pos = [(self.vehicles.get_absolute_position(veh_id) % self.scenario.length) / self.scenario.length
                          for veh_id in self.sorted_ids]
        scaled_pos = [self.vehicles.get_absolute_position(veh_id) / self.scenario.length for veh_id in self.sorted_ids]
        scaled_vel = [self.vehicles.get_absolute_position(veh_id) / self.env_params.get_additional_param("target_velocity")
                      for veh_id in self.sorted_ids]

        return np.array([[scaled_vel[i] + normal(0, self.observation_vel_std),
                          scaled_pos[i] + normal(0, self.observation_pos_std)]
                         for i in range(len(self.sorted_ids))])

        # # for stabilizing the ring: place the rl car at index 0 to maintain continuity between rollouts
        # indx_rl = [ind for ind in range(len(self.sorted_ids)) if self.sorted_ids[ind] in self.rl_ids][0]
        # indx_sorted_ids = np.mod(np.arange(len(self.sorted_ids)) + indx_rl, len(self.sorted_ids))
        #
        # return np.array([[scaled_vel[i] + normal(0, self.observation_vel_std),
        #                   scaled_pos[i] + normal(0, self.observation_pos_std)]
        #                  for i in indx_sorted_ids])

    # def render(self):
    #     print('current state/velocity:', self.state)


class SimpleMultiAgentAccelerationEnvironment(SimpleAccelerationEnvironment):

    @property
    def action_space(self):
        """
        Actions are a set of accelerations from 0 to 15m/s
        :return:
        """
        action_space = []
        for veh_id in self.rl_ids:
            action_space.append(Box(low=self.env_params.get_additional_param("max-deacc"),
                high=self.env_params.get_additional_param("max-acc"), shape=(1, )))
        return action_space

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        num_vehicles = self.scenario.num_vehicles
        observation_space = []
        speed = Box(low=0, high=np.inf, shape=(num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(num_vehicles,))
        #dist_to_intersection = Box(low=-np.inf, high=np.inf, shape=(self.scenario.num_vehicles,))
        obs_tuple = Tuple((speed, absolute_pos))
        for veh_id in self.rl_ids:
            observation_space.append(obs_tuple)
        return observation_space

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        return multi_agent_rewards.desired_velocity(
            state, rl_actions, fail=kwargs["fail"], target_velocity=self.env_params.get_additional_param("target_velocity"))

    def get_state(self, **kwargs):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        obs_arr = []
        for i in range(self.scenario.num_rl_vehicles):
            speed = [self.vehicles[veh_id]["speed"] for veh_id in self.sorted_ids]
            abs_pos = [self.vehicles[veh_id]["absolute_position"] for veh_id in self.sorted_ids]
            tup = (speed, abs_pos)
            obs_arr.append(tup)

        return obs_arr


class SimplePartiallyObservableEnvironment(SimpleAccelerationEnvironment):
    """
    This environment is an extension of the SimpleAccelerationEnvironment (seen above), with the exception
    that only partial information is provided to the agent about the network; namely, information on the
    vehicle immediately in front of it and the vehicle immediately behind it. The reward function, however,
    continues to reward GLOBAL performance. The environment also assumes that only one autonomous vehicle
    is in the network.
    """
    # TODO: maybe generalize for several vehicles (n-cars ahead, n-cars behind)?

    @property
    def action_space(self):
        """
        Actions are a set of accelerations from 0 to 15m/s
        :return:
        """
        return Box(low=-np.abs(self.env_params.get_additional_param("max-deacc")), high=self.env_params.get_additional_param("max-acc"),
                   shape=(self.scenario.num_rl_vehicles, ))

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        # self.obs_var_labels = ["Velocity", "Relative_pos"]
        speed = Box(low=0, high=np.inf, shape=(3,))
        absolute_pos = Box(low=0., high=np.inf, shape=(3,))
        return Tuple([speed, absolute_pos])

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        return rewards.desired_velocity(self, fail=kwargs["fail"])

    def get_state(self, **kwargs):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        vehID = self.rl_ids[0]
        lead_id = self.vehicles[vehID]["leader"]
        trail_id = self.vehicles[vehID]["follower"]
        max_speed = 30

        if trail_id is None:
            trail_id = vehID
            self.vehicles[trail_id]["headway"] = 0
        if lead_id is None:
            lead_id = vehID
            self.vehicles[vehID]["headway"] = 0

        # state contains the speed of the rl car, and its leader and follower,
        # as well as the rl car's position in the network, and its headway with the vehicles adjacent to it
        observation = np.array([
            [self.vehicles[vehID]["speed"] / max_speed,
             self.vehicles[vehID]["absolute_position"] / self.scenario.length],
            [self.vehicles[trail_id]["speed"] / max_speed,
             self.vehicles[trail_id]["headway"] / self.scenario.length],
            [self.vehicles[lead_id]["speed"] / max_speed,
             self.vehicles[vehID]["headway"] / self.scenario.length]])

        return observation
