import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flow.controllers.base_controller import BaseController
from replay_buffer import ReplayBuffer


class ImitatingController(BaseController):
    """
    Controller which uses a given neural net to imitate an expert. Subclasses BaseController
    """
    # Implementation in Tensorflow

    def __init__(self, veh_id, action_network, multiagent, car_following_params=None, time_delay=0.0, noise=0, fail_safe=None):

        BaseController.__init__(self, veh_id, car_following_params, delay=time_delay, fail_safe=fail_safe, noise=noise)
        self.action_network = action_network
        self.multiagent = multiagent
        self.veh_id = veh_id

    def get_accel(self, env):
        """
        Get acceleration for vehicle in the env
        """

        if self.multiagent:
            observation = env.get_state()[self.veh_id]
        else:
            observation = env.get_state()

        action = self.action_network.get_accel_from_observation(observation)

        if not self.multiagent:
            if self.action_network.action_dim > 1:
                # TODO: fill in
                try:
                    rl_ids = env.get_sorted_rl_ids()
                except:
                    print("Error caught: no get_sorted_rl_ids function, using get_rl_ids instead")
                    rl_ids = env.get_rl_ids()

                assert self.veh_id in rl_ids, "Vehicle corresponding to controller not in env!"

                ind = list.index(self.veh_id)
                return action[ind]
