import numpy as np
import tensorflow as tf
from utils import *
import tensorflow_probability as tfp
from flow.controllers.base_controller import BaseController
from replay_buffer import ReplayBuffer


class ImitatingController(BaseController):
    """
    Controller which learns to imitate another given expert controller.
    """
    # Implementation in Tensorflow

    def __init__(self, veh_id, action_network, multiagent, car_following_params=None, time_delay=0.0, noise=0, fail_safe=None):

        BaseController.__init__(self, veh_id, car_following_params, delay=time_delay, fail_safe=fail_safe, noise=noise)
        self.action_network = action_network
        self.multiagent = multiagent

    def get_accel(self, env):
        if self.multiagent:
            observation = env.get_state()[self.veh_id]
        else:
            observation = env.get_state()

        return self.action_network.get_accel_from_observation(observation)
