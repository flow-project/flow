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
        """
        Args:
            veh_id: ID of vehicle to control
            action_network: Instance of imitating_network class; neural net that gives action given state
            multiagent: boolean indicating if env is multiagent or singleagent
        """

        BaseController.__init__(self, veh_id, car_following_params, delay=time_delay, fail_safe=fail_safe, noise=noise)
        self.action_network = action_network  # neural network which specifies action to take
        self.multiagent = multiagent # whether env is multiagent or singleagent
        self.veh_id = veh_id # vehicle id that controller is controlling

    def get_accel(self, env):
        """
        Args:
            env: instance of environment being used

        Get acceleration for vehicle in the env, using action_network. Overrides superclass method.
        """
        # observation is a dictionary for multiagent envs, list for singleagent envs
        if self.multiagent:
            observation = env.get_state()[self.veh_id]
        else:
            observation = env.get_state()

        # get action from neural net
        action = self.action_network.get_accel_from_observation(observation)[0]

        # handles singleagent case in which there are multiple RL vehicles sharing common state
        # if action space is multidimensional, obtain the corresponding action for the vehicle
        if not self.multiagent and self.action_network.action_dim > 1:

            # get_sorted_rl_ids used for singleagent_straight_road; use get_rl_ids if method does not exist
            try:
                rl_ids = env.get_sorted_rl_ids()
            except:
                print("Error caught: no get_sorted_rl_ids function, using get_rl_ids instead")
                rl_ids = env.k.vehicle.get_rl_ids()

            assert self.veh_id in rl_ids, "Vehicle corresponding to controller not in env!"

            # return the action taken by the vehicle
            ind = rl_ids.index(self.veh_id)
            return action[ind]

        # in other cases, acceleration is the output of the network
        return action[0]
