import numpy as np
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

from flow.core import rewards
from flow.envs.intersection_env import IntersectionEnvironment


class TwoIntersectionEnvironment(IntersectionEnvironment):
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
        return Box(low=-np.abs(self.env_params.max_deacc),
                   high=self.env_params.max_acc,
                   shape=(self.vehicles.num_rl_vehicles, ))

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        speed = Box(low=0, high=np.inf, shape=(self.vehicles.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))
        return Tuple((speed, absolute_pos))

    def apply_rl_actions(self, rl_actions):
        """
        See parent class
        """
        self.sorted_ids = self.sort_by_intersection_dist()
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]
        for i, veh_id in enumerate(sorted_rl_ids):
            this_speed = self.vehicles.get_speed(veh_id)
            enter_speed = self.scenario.initial_config.additional_params["enter_speed"]

            # If we are outside the control region, just accelerate
            # up to the entering velocity
            if self.get_distance_to_intersection(veh_id)[0] > 50 or self.get_distance_to_intersection(veh_id)[0] < 0:
                # get up to max speed
                if this_speed < enter_speed:
                    # accelerate as fast as you are allowed
                    if ((enter_speed - this_speed)/self.time_step > self.env_params.max_acc):
                        rl_actions[i] =  self.env_params.max_acc
                    # accelerate the exact amount needed to get up to target velocity
                    else:
                        rl_actions[i] = ((enter_speed - this_speed)/self.time_step)
                # at max speed, don't accelerate
                else:
                    rl_actions[i] = 0.0
            # we are at the intersection, turn the fail-safe on
            # make it so this only happens once
            else:
                self.traci_connection.vehicle.setSpeedMode(veh_id, 1)
            # cap the velocity 
            if this_speed + self.time_step*rl_actions[i] > enter_speed:
                rl_actions[i] = ((enter_speed - this_speed)/self.time_step)

        self.apply_acceleration(sorted_rl_ids, rl_actions)

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
        length = self.scenario.net_params.additional_params["length"]
        enter_speed = self.scenario.initial_config.additional_params["enter_speed"]
        return np.array([[self.vehicles.get_speed(veh_id)/enter_speed,
                          self.vehicles.get_absolute_position(veh_id)/length]
                         for veh_id in self.sorted_ids])
