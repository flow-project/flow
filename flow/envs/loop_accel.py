from flow.envs.base_env import SumoEnvironment
from flow.core import rewards
from flow.core import multi_agent_rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np

class SimpleAccelerationEnvironment(SumoEnvironment):
    """
    Fully functional environment for single lane closed loop settings. Takes in
    an *acceleration* as an action. Reward function is negative norm of the
    difference between the velocities of each vehicle, and the target velocity.
    State function is a vector of the velocities and absolute positions for each
    vehicle.
    """

    @property
    def action_space(self):
        """
        See parent class

        Actions are a set of accelerations from max-deacc to max-acc for each
        rl vehicle.
        """
        return Box(low=-np.abs(self.env_params.max_deacc),
                   high=self.env_params.max_acc,
                   shape=(self.vehicles.num_rl_vehicles, ))

    @property
    def observation_space(self):
        """
        See parent class

        An observation is an array the velocities and absolute positions for
        each vehicle
        """
        self.obs_var_labels = ["Velocity", "Absolute_pos"]
        speed = Box(low=0, high=np.inf, shape=(self.vehicles.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))
        return Tuple((speed, absolute_pos))

    def apply_rl_actions(self, rl_actions):
        """
        See parent class

        Accelerations are applied to rl vehicles in accordance with the commands
        provided by rllab. These actions may be altered by flow's failsafes or
        sumo-defined speed modes.
        """
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]
        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        # reward desired velocity
        reward = rewards.desired_velocity(self, fail=kwargs["fail"])

        return reward

    def get_state(self, **kwargs):
        """
        See parent class

        The state is an array of velocities and absolute positions for each
        vehicle
        """
        scaled_pos = [self.vehicles.get_absolute_position(veh_id) /
                      self.scenario.length for veh_id in self.sorted_ids]
        scaled_vel = [self.vehicles.get_speed(veh_id) /
                      self.env_params.get_additional_param("target_velocity")
                      for veh_id in self.sorted_ids]

        return np.array([[scaled_vel[i], scaled_pos[i]]
                         for i in range(len(self.sorted_ids))])

    def get_leader_blocker_headways(self, ego_id):
        """
        returns a list of headways for each lane
        :param ego_id: the ego vehicle
        :return: array of headways
        """
        curr_pos = self.get_x_by_id(ego_id)
        min_headways = [float("inf")] * self.scenario.lanes
        min_reverse_headways = [float("inf")] * self.scenario.lanes

        for veh_id in self.ids:
            if veh_id != ego_id:
                lane = self.vehicles.get_lane(veh_id)
                min_headways[lane] = min((self.get_x_by_id(veh_id) - curr_pos) % self.scenario.length, min_headways[lane])
                min_reverse_headways[lane] = min((curr_pos - self.get_x_by_id(veh_id)) % self.scenario.length, min_reverse_headways[lane])

        # print(min_headways, min_reverse_headways)
        return min_headways, min_reverse_headways

class SimpleMultiAgentAccelerationEnvironment(SimpleAccelerationEnvironment):
    """
    An extension of SimpleAccelerationEnvironment which treats each autonomous
    vehicles as a separate rl agent, thereby allowing autonomous vehicles to be
    trained in multi-agent settings.
    """

    @property
    def action_space(self):
        """
        See parent class

        Actions are a set of accelerations from max-deacc to max-acc for each
        rl vehicle.
        """
        action_space = []
        for veh_id in self.rl_ids:
            action_space.append(Box(low=self.env_params.max_deacc,
                high=self.env_params.max_acc, shape=(1, )))
        return action_space

    @property
    def observation_space(self):
        """
        See parent class
        """
        num_vehicles = self.scenario.num_vehicles
        observation_space = []
        speed = Box(low=0, high=np.inf, shape=(num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(num_vehicles,))
        obs_tuple = Tuple((speed, absolute_pos))
        for veh_id in self.rl_ids:
            observation_space.append(obs_tuple)
        return observation_space

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        return multi_agent_rewards.desired_velocity(
            state, rl_actions,
            fail=kwargs["fail"],
            target_velocity=self.env_params.get_additional_param("target_velocity"))

    def get_state(self, **kwargs):
        """
        See parent class
        The state is an array the velocities and absolute positions for
        each vehicle.
        """
        obs_arr = []
        for i in range(self.scenario.num_rl_vehicles):
            speed = [self.vehicles.get_speed(veh_id)
                     for veh_id in self.sorted_ids]
            abs_pos = [self.vehicles.get_absolute_position(veh_id)
                       for veh_id in self.sorted_ids]
            tup = (speed, abs_pos)
            obs_arr.append(tup)

        return obs_arr

class SimplePartiallyObservableEnvironment(SimpleAccelerationEnvironment):
    """
    This environment is an extension of the SimpleAccelerationEnvironment, with
    the exception that only local information is provided to the agent about the
    network; i.e. headway, velocity, and velocity difference. The reward
    function, however, continues to reward global network performance.

    NOTE: The environment also assumes that there is only one autonomous vehicle
    is in the network.
    """

    @property
    def observation_space(self):
        """
        See parent class
        """
        return Box(low=-np.inf, high=np.inf, shape=(3,))

    def get_state(self, **kwargs):
        """
        See parent class

        The state is an array consisting of the speed of the rl vehicle, the
        relative speed of the vehicle ahead of it, and the headway between the
        rl vehicle and the vehicle ahead of it.
        """
        rl_id = self.rl_ids[0]
        lead_id = self.vehicles[rl_id]["leader"]
        max_speed = self.max_speed

        # if a vehicle crashes into the car ahead of it, it no longer processes
        # a lead vehicle
        if lead_id is None:
            lead_id = rl_id
            self.vehicles[rl_id]["headway"] = 0

        observation = np.array([
            [self.vehicles[rl_id]["speed"] / max_speed],
            [(self.vehicles[lead_id]["speed"] - self.vehicles[rl_id]["speed"])
             / max_speed],
            [self.vehicles[rl_id]["headway"] / self.scenario.length]])

        return observation
