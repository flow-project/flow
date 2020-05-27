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

    def load_network(self, load_path):
        if self.stochastic:
            self.model = tf.keras.models.load_model(load_path, custom_objects={'negative_log_likelihood_loss': negative_log_likelihood_loss})


    def save_network_PPO(self, save_path):
        input = tf.keras.layers.Input(self.model.input.shape[1].value)
        curr_layer = input

        # number of hidden layers
        num_layers = len(self.model.layers) - 2

        # build layers for policy
        for i in range(num_layers):
            size = self.model.layers[i + 1].output.shape[1].value
            activation = tf.keras.activations.serialize(self.model.layers[i + 1].activation)
            curr_layer = tf.keras.layers.Dense(size, activation=activation, name="policy_hidden_layer_{}".format(i + 1))(curr_layer)
        output_layer_policy = tf.keras.layers.Dense(self.model.output.shape[1].value, activation=None, name="policy_output_layer")(curr_layer)

        # build layers for value function
        curr_layer = input
        for i in range(num_layers):
            curr_layer = tf.keras.layers.Dense(self.size, activation="tanh", name="vf_hidden_layer_{}".format(i+1))(curr_layer)
        output_layer_vf = tf.keras.layers.Dense(1, activation=None, name="vf_output_layer")(curr_layer)

        ppo_model = tf.keras.Model(inputs=input, outputs=[output_layer_policy, output_layer_vf], name="ppo_model")

        # set the policy weights to those learned from imitation
        for i in range(num_layers):
            policy_layer = ppo_model.get_layer(name="policy_hidden_layer_{}".format(i + 1))
            policy_layer.set_weights(self.model.layers[i + 1].get_weights())
        policy_output = ppo_model.get_layer("policy_output_layer")
        policy_output.set_weights(self.model.layers[-1].get_weights())

        # save the model (as a h5 file)
        ppo_model.save(save_path)









