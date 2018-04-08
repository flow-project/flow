import numpy as np
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

from flow.core import rewards
from flow.envs.intersection_env import IntersectionEnv


class TwoIntersectionEnv(IntersectionEnv):
    """Environment for training autonomous vehicles in a two-way intersection
    scenario.

    States
    ------
    (blank)

    Actions
    -------
    (blank)

    Rewards
    -------
    (blank)

    Termination
    -----------
    (blank)
    """

    @property
    def action_space(self):
        return Box(low=-np.abs(self.env_params.max_decel),
                   high=self.env_params.max_accel,
                   shape=(self.vehicles.num_rl_vehicles,),
                   dtype=np.float32)

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        speed = Box(low=0, high=np.inf, shape=(self.vehicles.num_vehicles,),
                    dtype=np.float32)
        absolute_pos = Box(low=0., high=np.inf,
                           shape=(self.vehicles.num_vehicles,),
                           dtype=np.float32)
        return Tuple((speed, absolute_pos))

    def _apply_rl_actions(self, rl_actions):
        """
        See parent class
        """
        self.sorted_ids = self.sort_by_intersection_dist()
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]
        init_config = self.scenario.initial_config
        for i, veh_id in enumerate(sorted_rl_ids):
            this_speed = self.vehicles.get_speed(veh_id)
            enter_speed = init_config.additional_params["enter_speed"]

            # If we are outside the control region, just accelerate
            # up to the entering velocity
            if self.get_distance_to_intersection(veh_id)[0] > 50 or \
                    self.get_distance_to_intersection(veh_id)[0] < 0:
                # get up to max speed
                if this_speed < enter_speed:
                    # accelerate as fast as you are allowed
                    speed_diff = enter_speed - this_speed
                    if ((speed_diff) / self.sim_step > self.env_params.max_acc):
                        rl_actions[i] = self.env_params.max_acc
                    # accelerate the exact amount needed to target velocity
                    else:
                        rl_actions[i] = ((speed_diff) / self.sim_step)
                # at max speed, don't accelerate
                else:
                    rl_actions[i] = 0.0
            # we are at the intersection, turn the fail-safe on
            # make it so this only happens once
            else:
                self.traci_connection.vehicle.setSpeedMode(veh_id, 1)
            # cap the velocity 
            if this_speed + self.sim_step * rl_actions[i] > enter_speed:
                rl_actions[i] = ((enter_speed - this_speed) / self.sim_step)

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
        init_config = self.scenario.initial_config
        enter_speed = init_config.additional_params["enter_speed"]
        return np.array([[self.vehicles.get_speed(veh_id) / enter_speed,
                          self.vehicles.get_absolute_position(veh_id) / length]
                         for veh_id in self.sorted_ids])
