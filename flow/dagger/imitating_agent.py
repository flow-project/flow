import numpy as np
import tensorflow as tf
import time
from imitating_controller import *
from replay_buffer

class Imitating_Agent(object):
    def __init__(self, sess, env, params):
        self.env = env
        self.sess = sess
        self.params = params

        self.policy = Imitator_Policy(sess, self.params['action_dim'], self.params['obs_dim'], self.params['num_layers'], self.params['size'], self.params['learning_rate'])

        self.replay_buffer = ReplayBuffer(self.params['replay_buffer_size'])


    def train(self, obs, acts):
        self.policy.update(obs, acts)

    def add_to_replay_buffer(self, rollout_list):
        self.replay_buffer.add_rollouts(rollout_list)

    def sample_data(self, batch_size):
        return self.replay_buffer.sample_batch(batch_size)
