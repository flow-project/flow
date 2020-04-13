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

    def __init__(self, veh_id, sess, action_dim, obs_dim, num_layers, size, learning_rate, replay_buffer_size, training = True, inject_noise=0, noise_variance=0.5, policy_scope='policy_vars', car_following_params=None, time_delay=0.0, noise=0, fail_safe=None):

        BaseController.__init__(self, veh_id, car_following_params, delay=time_delay, fail_safe=fail_safe, noise=noise)
        self.sess = sess
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_layers = num_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.inject_noise=inject_noise
        self.noise_variance = noise_variance

        with tf.variable_scope(policy_scope, reuse=tf.AUTO_REUSE):
            self.build_network()


        if self.training:
            self.replay_buffer = ReplayBuffer(replay_buffer_size)
        else:
            self.replay_buffer = None

        self.policy_vars = [v for v in tf.all_variables() if policy_scope in v.name and 'train' not in v.name]
        self.saver = tf.train.Saver(self.policy_vars, max_to_keep=None)

    def build_network(self):
        """
        Defines neural network for choosing actions.
        """
        self.define_placeholders()
        self.define_forward_pass()
        if self.training:
            with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
                self.define_train_op()


    def define_placeholders(self):
        """
        Defines input, output, and training placeholders for neural net
        """
        self.obs_placeholder = tf.placeholder(shape=[None, self.obs_dim], name="obs", dtype=tf.float32)
        self.action_placeholder = tf.placeholder(shape=[None, self.action_dim], name="action", dtype=tf.float32)

        if self.training:
            self.action_labels_placeholder = tf.placeholder(shape=[None, self.action_dim], name="labels", dtype=tf.float32)

    def define_forward_pass(self):
        pred_action = build_neural_net(self.obs_placeholder, output_size=self.action_dim, scope='network_scope', n_layers=self.num_layers, size=self.size)
        self.action_predictions = pred_action
        print("TYPE: ", type(self.obs_placeholder))

        if self.inject_noise == 1:
            self.action_predictions = self.action_predictions + tf.random_normal(tf.shape(self.action_predictions), 0, self.noise_variance)

    def define_train_op(self):
        true_actions = self.action_labels_placeholder
        predicted_actions = self.action_predictions

        self.loss = tf.losses.mean_squared_error(true_actions, predicted_actions)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, observation_batch, action_batch):
        action_batch = action_batch.reshape(action_batch.shape[0], self.action_dim)
        ret = self.sess.run([self.train_op, self.loss], feed_dict={self.obs_placeholder: observation_batch, self.action_labels_placeholder: action_batch})

    def get_accel_from_observation(self, observation):
        # network expects an array of arrays (matrix); if single observation (no batch), convert to array of arrays
        if len(observation.shape)<=1:
            observation = observation[None]
        ret_val = self.sess.run([self.action_predictions], feed_dict={self.obs_placeholder: observation})[0]

        return ret_val

    def get_accel(self, env):
        # network expects an array of arrays (matrix); if single observation (no batch), convert to array of arrays
        observation = env.get_state()
        return self.get_accel_from_observation(observation)

    def add_to_replay_buffer(self, rollout_list):
        """ Add rollouts to replay buffer """

        self.replay_buffer.add_rollouts(rollout_list)

    def sample_data(self, batch_size):
        """ Sample a batch of data from replay buffer """

        return self.replay_buffer.sample_batch(batch_size)

    def save_network(self, save_path):
        self.saver.save(self.sess, save_path)
