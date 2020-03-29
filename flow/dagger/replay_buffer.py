import time
import numpy as np
import tensorflow as tf
import gym
import os
from utils import *


class ReplayBuffer(object):
    def __init__(self, max_size=100000):

        self.max_size = max_size

        # store each rollout
        self.rollouts = []

        # store component arrays from each rollout
        self.observations = None
        self.actions = None
        self.expert_actions = None
        self.rewards = None
        self.next_observations = None
        self.terminals = None


    def add_rollouts(self, rollouts_list):
        """
            Add a list of rollouts to the replay buffer
        """

        for rollout in rollouts_list:
            self.rollouts.append(rollout)

        observations, actions, expert_actions, rewards, next_observations, terminals = unpack_rollouts(rollouts_list)
        assert (not np.any(np.isnan(expert_actions))), "REPLAY BUFFER ERROR"

        if self.observations is None:
            self.observations = observations[-self.max_size:]
            self.actions = actions[-self.max_size:]
            self.expert_actions = expert_actions[-self.max_size:]
            self.rewards = rewards[-self.max_size:]
            self.next_observations = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.observations = np.concatenate([self.observations, observations])[-self.max_size:]
            self.actions = np.concatenate([self.actions, actions])[-self.max_size:]
            self.expert_actions = np.concatenate([self.expert_actions, expert_actions])[-self.max_size:]
            self.rewards = np.concatenate([self.rewards, rewards])[-self.max_size:]
            self.next_observations = np.concatenate([self.next_observations, next_observations])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]

    def sample_batch(self, batch_size):
        """
            Sample a batch of data (with size batch_size) from replay buffer.
            Returns data in separate numpy arrays of observations, actions, rewards, next_observations, terminals
        """
        assert self.observations is not None and self.actions is not None and self.expert_actions is not None and self.rewards is not None and self.next_observations is not None and self.terminals is not None

        size = len(self.observations)
        rand_inds = np.random.randint(0, size, batch_size)
        return self.observations[rand_inds], self.actions[rand_inds], self.expert_actions[rand_inds], self.rewards[rand_inds], self.next_observations[rand_inds], self.terminals[rand_inds]
