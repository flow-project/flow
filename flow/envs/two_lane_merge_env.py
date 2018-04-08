from flow.envs.loop_accel import AccelMAEnv
from flow.core import multi_agent_rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np

class TwoLaneStraightMerge(AccelMAEnv):
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
        for _ in self.vehicles.get_rl_ids():
            action_space.append(Box(low=self.env_params.max_deacc,
                                    high=self.env_params.max_acc,
                                    shape=(1, )))
        return action_space

    @property
    def observation_space(self):
        """
        See parent class
        """
        num_vehicles = self.vehicles.num_vehicles
        observation_space = []
        speed = Box(low=0, high=np.inf, shape=(num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(num_vehicles,))
        obs_tuple = Tuple((speed, absolute_pos))
        for _ in self.vehicles.get_rl_ids():
            observation_space.append(obs_tuple)
        return observation_space

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        return multi_agent_rewards.desired_velocity(
            state, rl_actions,
            fail=kwargs["fail"],
            target_velocity=self.env_params.additional_params["target_velocity"]
        )

    def get_state(self, **kwargs):
        """
        See parent class
        The state is an array the velocities and absolute positions for
        each vehicle.
        """
        obs_arr = []
        for rl_id in self.rl_ids:
            # Re-sort based on the rl agent being in front
            # Probably should try and do this less often
            sorted_indx = np.argsort(
                [(self.vehicles.get_absolute_position(veh_id) -
                  self.vehicles.get_absolute_position(rl_id))
                 % self.scenario.length for veh_id in self.ids])
            sorted_ids = np.array(self.ids)[sorted_indx]

            speed = [self.vehicles.get_speed(veh_id) for veh_id in sorted_ids]
            abs_pos = [(self.vehicles.get_absolute_position(veh_id) -
                        self.vehicles.get_absolute_position(rl_id))
                       % self.scenario.length for veh_id in sorted_ids]

            tup = (speed, abs_pos)
            obs_arr.append(tup)

        return obs_arr