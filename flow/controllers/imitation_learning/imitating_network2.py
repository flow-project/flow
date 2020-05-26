import numpy as np
import tensorflow as tf
from utils_tensorflow import *
from keras_utils import *
import tensorflow_probability as tfp
from flow.controllers.base_controller import BaseController
from replay_buffer import ReplayBuffer


class ImitatingNetwork2():
    """
    Class containing neural network which learns to imitate a given expert controller.
    """

    def __init__(self, sess, action_dim, obs_dim, num_layers, size, learning_rate, replay_buffer_size, training = True, stochastic=False, policy_scope='policy_vars', load_existing=False, load_path=''):

        """
        Initializes and constructs neural network

        Args:
            sess: Tensorflow session variable
            action_dim: dimension of action space (determines size of network output)
            obs_dim: dimension of observation space (size of network input)
            num_layers: number of hidden layers (for an MLP)
            size: size of each layer in network
            learning_rate: learning rate used in optimizer
            replay_buffer_size: maximum size of replay buffer used to hold data for training
            training: boolean, whether the network will be trained (as opposed to loaded)
            stochastic: boolean indicating if the network outputs a stochastic (multivariate Gaussian) or deterministic policy
            policy_scope: variable scope used by Tensorflow for weights/biases
            load_existing: boolean, whether to load an existing tensorflow model
            load_path: path to directory containing an existing tensorflow model

        """

        self.sess = sess
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_layers = num_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.stochastic=stochastic

        print("INNNNNITITTTTT")

        # load network if specified, or construct network
        if load_existing:
            self.load_network(load_path)

        else:
            self.build_network()
            self.compile_network()


        # init replay buffer
        if self.training:
            self.replay_buffer = ReplayBuffer(replay_buffer_size)
        else:
            self.replay_buffer = None


    def build_network(self):
        """
        Defines neural network for choosing actions. Defines placeholders and forward pass
        """
        # setup placeholders for network input and labels for training, and hidden layers/output
        if self.stochastic:
            self.model = build_neural_net_stochastic(self.obs_dim, self.action_dim, self.num_layers, self.size)
        else:
            self.model = build_neural_net_deterministic(self.obs_dim, self.action_dim, self.num_layers, self.size)


    def compile_network(self):
        loss = get_loss(self.stochastic)
        self.model.compile(loss=loss, optimizer='adam')


    def train(self, observation_batch, action_batch):
        """
        Executes one training step for the given batch of observation and action data
        """
        # reshape action_batch to ensure a shape (batch_size, action_dim)
        action_batch = action_batch.reshape(action_batch.shape[0], self.action_dim)
        batch_size = action_batch.shape[0]
        self.model.fit(observation_batch, action_batch, batch_size=batch_size, epochs=1, steps_per_epoch=1)

    def get_accel_from_observation(self, observation):
        """
        Gets the network's acceleration prediction based on given observation/state
        """

        # network expects an array of arrays (matrix); if single observation (no batch), convert to array of arrays
        if len(observation.shape)<=1:
            observation = observation[None]
        # "batch size" is 1, so just get single acceleration/acceleration vector
        network_output = self.model.predict(observation)
        if self.stochastic:
            mean, log_std = network_output[:, :self.action_dim], network_output[:, self.action_dim:]
            var = np.exp(2 * log_std)
            action = np.random.multivariate_normal(mean[0], var)
            return action
        else:
            return network_output

    def get_accel(self, env):
        """
        Get network's acceleration prediction(s) based on given env
        """
        observation = env.get_state()
        return self.get_accel_from_observation(observation)


    def add_to_replay_buffer(self, rollout_list):
        """ Add rollouts to replay buffer """

        self.replay_buffer.add_rollouts(rollout_list)


    def sample_data(self, batch_size):
        """ Sample a batch of data from replay buffer """

        return self.replay_buffer.sample_batch(batch_size)

    def save_network(self, save_path):
        """ Save network to given path and to tensorboard """

        self.model.save(save_path)
        # tensorboard

        # writer = tf.summary.FileWriter('./graphs2', tf.get_default_graph())
