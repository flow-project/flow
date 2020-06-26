import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flow.controllers.base_controller import BaseController
from flow.controllers.imitation_learning.replay_buffer import ReplayBuffer

class ImitatingController(BaseController):
    """
    Controller which uses a given neural net to imitate an expert. Subclasses BaseController
    """

    # Implementation in Tensorflow Keras

    def __init__(self, veh_id, action_network, multiagent, car_following_params=None, time_delay=0.0, noise=0, fail_safe=None):
        """
        Parameters
        __________
        veh_id: String
            ID of vehicle to control
        action_network: ImitatingNetwork
            Instance of imitating_network class; neural net that gives action given state
        multiagent: bool
            boolean indicating if env is multiagent or singleagent
        """

        BaseController.__init__(self, veh_id, car_following_params, delay=time_delay, fail_safe=fail_safe, noise=noise)
        self.action_network = action_network
        self.multiagent = multiagent
        self.veh_id = veh_id


    def get_accel(self, env):
        """
        Get acceleration for vehicle in the environment. Overrides superclass method.
        Parameters
        __________
        env: Gym Env
            instance of environment being used
        """
        # observation is a dictionary for multiagent envs, list for singleagent envs

        if self.multiagent:
            # if vehicle is in non-control edge, it will not be in observation, so return None to default control to Sumo
            if self.veh_id not in env.get_state().keys():
                return None
            observation = env.get_state()[self.veh_id]
        else:
            observation = env.get_state()

        # get action from neural net
        action = self.action_network.get_accel_from_observation(observation)[0]

        # handles singleagent case in which there are multiple RL vehicles sharing common state
        # if action space is multidimensional, obtain the corresponding action for the vehicle
        if not self.multiagent and self.action_network.action_dim > 1:

            # get_sorted_rl_ids used for singleagent_straight_road; use get_rl_ids if method does not exist
            if hasattr(env, 'get_sorted_rl_ids'):
                rl_ids = env.get_sorted_rl_ids()
            else:
                rl_ids = env.get_rl_ids()

            if not (self.veh_id in rl_ids):
                # vehicle in non-control edge, so return None to default control to Sumo
                return None 

            # return the action taken by the vehicle
            ind = rl_ids.index(self.veh_id)
            return action[ind]

        # in other cases, acceleration is the output of the network
        return action
