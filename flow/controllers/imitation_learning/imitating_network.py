import numpy as np
import tensorflow as tf
from utils_tensorflow import *
from keras_utils import *
import tensorflow_probability as tfp
from flow.controllers.base_controller import BaseController
from replay_buffer import ReplayBuffer
from time import time
from tensorflow.python.keras.callbacks import TensorBoard


class ImitatingNetwork():
    """
    Class containing neural network which learns to imitate a given expert controller.
    """

    def __init__(self, sess, action_dim, obs_dim, fcnet_hiddens, replay_buffer_size, stochastic=False, variance_regularizer = 0, load_model=False, load_path=''):

        """Initializes and constructs neural network.
        Parameters
        ----------
        sess : tf.Session
            Tensorflow session variable
        action_dim : int
            action_space dimension
        obs_dim : int
            dimension of observation space (size of network input)
        fcnet_hiddens : list
            list of hidden layer sizes for fully connected network (length of list is number of hidden layers)
        replay_buffer_size: int
            maximum size of replay buffer used to hold data for training
        stochastic: bool
            indicates if network outputs a stochastic (MV Gaussian) or deterministic policy
        variance_regularizer: float
            regularization hyperparameter to penalize high variance policies
        load_model: bool
            if True, load model from path specified in load_path
        load_path: String
            path to h5 file containing model to load.

        """

        self.sess = sess
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.fcnet_hiddens = fcnet_hiddens
        self.stochastic=stochastic
        self.variance_regularizer = variance_regularizer

        # load network if specified, or construct network
        if load_model:
            self.load_network(load_path)

        else:
            self.build_network()
            self.compile_network()

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def build_network(self):
        """
        Defines neural network for choosing actions. Defines placeholders and forward pass
        """
        # setup placeholders for network input and labels for training, and hidden layers/output
        if self.stochastic:
            self.model = build_neural_net_stochastic(self.obs_dim, self.action_dim, self.fcnet_hiddens)
        else:
            self.model = build_neural_net_deterministic(self.obs_dim, self.action_dim, self.fcnet_hiddens)


    def compile_network(self):
        """
        Compiles Keras network with appropriate loss and optimizer
        """
        loss = get_loss(self.stochastic, self.variance_regularizer)
        self.model.compile(loss=loss, optimizer='adam')


    def train(self, observation_batch, action_batch):
        """
        Executes one training (gradient) step for the given batch of observation and action data

        Parameters
        ----------
        observation_batch : numpy array
            numpy array containing batch of observations (inputs)
        action_batch : numpy array
            numpy array containing batch of actions (labels)
        """

        # reshape action_batch to ensure a shape (batch_size, action_dim)
        action_batch = action_batch.reshape(action_batch.shape[0], self.action_dim)
        # one gradient step on batch
        self.model.train_on_batch(observation_batch, action_batch)

    def get_accel_from_observation(self, observation):
        """
        Gets the network's acceleration prediction based on given observation/state

        Parameters
        ----------
        observation : numpy array
            numpy array containing a single observation

        Returns
        -------
        numpy array
            one element numpy array containing accleeration
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

        Parameters
        ----------
        env :
            environment object

        Returns
        -------
        numpy array
            one element numpy array containing accleeration

        """
        observation = env.get_state()
        return self.get_accel_from_observation(observation)


    def add_to_replay_buffer(self, rollout_list):
        """
        Add data to a replay buffer

        Parameters
        ----------
        rollout_list : list
            list of rollout dictionaries
        """

        self.replay_buffer.add_rollouts(rollout_list)


    def sample_data(self, batch_size):
        """
        Sample a batch of data from replay buffer.

        Parameters
        ----------
        batch_size : int
            size of batch to sample
        """

        return self.replay_buffer.sample_batch(batch_size)

    def save_network(self, save_path):
        """
        Save imitation network as a h5 file in save_path

        Parameters
        ----------
        save_path : String
            path to h5 file to save to
        """

        self.model.save(save_path)
        # tensorboard

        # writer = tf.summary.FileWriter('./graphs2', tf.get_default_graph())

    def load_network(self, load_path):
        """
        Load imitation network from a h5 file in load_path

        Parameters
        ----------
        load_path : String
            path to h5 file containing model to load from
        """
        if self.stochastic:
            self.model = tf.keras.models.load_model(load_path, custom_objects={'negative_log_likelihood_loss': negative_log_likelihood_loss})
        else:
            self.model = tf.keras.models.load_model(load_path)


    def save_network_PPO(self, save_path):
        """
        Build a model, with same policy architecture as imitation network, to run PPO, copy weights from imitation, and save this model.

        Parameters
        ----------
        load_path : save_path
            path to h5 file to save to
        """

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
            size = self.fcnet_hiddens[i]
            curr_layer = tf.keras.layers.Dense(size, activation="tanh", name="vf_hidden_layer_{}".format(i+1))(curr_layer)
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









