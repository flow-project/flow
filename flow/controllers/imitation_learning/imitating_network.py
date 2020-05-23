import numpy as np
import tensorflow as tf
from utils_tensorflow import *
import tensorflow_probability as tfp
from flow.controllers.base_controller import BaseController
from replay_buffer import ReplayBuffer


class ImitatingNetwork():
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

        # load network if specified, or construct network
        if load_existing:
            self.load_network(load_path)

        else:
            print("HERE")
            self.build_network()


        # init replay buffer
        if self.training:
            self.replay_buffer = ReplayBuffer(replay_buffer_size)
        else:
            self.replay_buffer = None

        # set up policy variables, and saver to save model. Save only non-training variables (weights/biases)
        if not load_existing:
            self.policy_vars = [v for v in tf.all_variables() if 'network_scope' in v.name and 'train' not in v.name]
            self.saver = tf.train.Saver(self.policy_vars, max_to_keep=None)

        # tensorboard
        self.writer = tf.summary.FileWriter('/Users/akashvelu/Documents/Random/tensorboard/', tf.get_default_graph())
        # track number of training steps
        self.train_steps = 0

    def build_network(self):
        """
        Defines neural network for choosing actions. Defines placeholders and forward pass
        """
        # setup placeholders for network input and labels for training, and hidden layers/output
        self.define_placeholders()
        self.define_forward_pass()
        # set up training operation (e.g. Adam optimizer)
        if self.training:
            with tf.variable_scope('train'):
                self.define_train_op()



    def load_network(self, path):
        """
        Load tensorflow model from the path specified, set action prediction to proper placeholder
        """
        # load and restore model
        loader = tf.train.import_meta_graph(path + 'model.ckpt.meta')
        loader.restore(self.sess, path+'model.ckpt')

        # get observation placeholder (for input into network)
        self.obs_placeholder = tf.get_default_graph().get_tensor_by_name('policy_vars/observation:0')
        # get output tensor (using name of appropriate tensor)
        network_output = tf.get_default_graph().get_tensor_by_name('policy_vars/network_scope/Output_Layer/BiasAdd:0')

        # for stochastic policies, the network output is twice the action dimension. First half specifies the mean of a multivariate gaussian distribution, second half specifies the diagonal entries for the diagonal covariance matrix.
        # for deterministic policies, network output is the action.
        if self.stochastic:
            # determine means and (diagonal entries of ) covariance matrices (could be many in the case of batch) for action distribution
            means = network_output[:, :self.action_dim]
            log_vars = network_output[:, self.action_dim:]
            vars = tf.math.exp(log_vars)

            # set up action distribution (parameterized by network output)
            # if a batch of size k is input as observations, then the the self.dist will store k different Gaussians
            self.dist = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=vars, name='Prediction Distribution')
            # action is a sample from this distribution; one sample output per Gaussian contained in self.dist
            self.action_predictions = self.dist.sample()
        else:
            self.dist = None
            self.action_predictions = network_output

    def define_placeholders(self):
        """
        Defines input, output, and training placeholders for neural net
        """
        # placeholder for observations (input into network)
        self.obs_placeholder = tf.placeholder(shape=[None, self.obs_dim], name="observation", dtype=tf.float32)

        # if training, define placeholder for labels (supervised learning)
        if self.training:
            self.action_labels_placeholder = tf.placeholder(shape=[None, self.action_dim], name="labels", dtype=tf.float32)


    def define_forward_pass(self):
        """
        Build network and initialize proper action prediction op
        """
        # network output is twice action dim if stochastic (1st half mean, 2nd half diagonal elements of covariance)
        if self.stochastic:
            output_size = 2 * self.action_dim
        else:
            output_size = self.action_dim

        # build forward pass and get the tensor for output of last layer
        network_output = build_neural_net(self.obs_placeholder, output_size=output_size, scope='network_scope', n_layers=self.num_layers, size=self.size)

        # parse the mean and covariance from output if stochastic, and set up distribution
        if self.stochastic:
            # determine means and (diagonal entries of ) covariance matrices (could be many in the case of batch) for action distribution

            means, log_vars = tf.split(network_output, num_or_size_splits=2, axis=1)
            vars = tf.math.exp(log_vars)

            # set up action distribution (parameterized by network output)
            # if a batch of size k is input as observations, then the the self.dist will store k different Gaussians
            with tf.variable_scope('Action_Distribution'):
                self.dist = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=vars)
            # action is a sample from this distribution; one sample output per Gaussian contained in self.dist
            self.action_predictions = self.dist.sample()

        else:
            self.dist = None
            self.action_predictions = network_output


    def define_train_op(self):
        """
        Defines training operations for network (loss function and optimizer)
        """
        # labels
        true_actions = self.action_labels_placeholder
        predicted_actions = self.action_predictions

        if self.stochastic:
            # negative log likelihood loss for stochastic policy
            self.loss = self.dist.log_prob(true_actions)
            self.loss = tf.negative(self.loss)
            self.loss = tf.reduce_mean(self.loss)
            summary_name = 'Loss_tracking_NLL'
        else:
            # MSE loss for deterministic policy
            self.loss = tf.losses.mean_squared_error(true_actions, predicted_actions)
            summary_name = 'Loss_tracking_MSE'


        self.loss_summary = tf.summary.scalar(name=summary_name, tensor=self.loss)
        # Adam optimizer
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, observation_batch, action_batch):
        """
        Executes one training step for the given batch of observation and action data
        """
        # reshape action_batch to ensure a shape (batch_size, action_dim)
        action_batch = action_batch.reshape(action_batch.shape[0], self.action_dim)
        _, loss, summary = self.sess.run([self.train_op, self.loss, self.loss_summary], feed_dict={self.obs_placeholder: observation_batch, self.action_labels_placeholder: action_batch})
        self.writer.add_summary(summary, global_step=self.train_steps)
        self.train_steps += 1

    def get_accel_from_observation(self, observation):
        """
        Gets the network's acceleration prediction based on given observation/state
        """

        # network expects an array of arrays (matrix); if single observation (no batch), convert to array of arrays
        if len(observation.shape)<=1:
            observation = observation[None]
        # "batch size" is 1, so just get single acceleration/acceleration vector
        ret_val = self.sess.run([self.action_predictions], feed_dict={self.obs_placeholder: observation})[0]
        return ret_val

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

        self.saver.save(self.sess, save_path)
        # tensorboard
        writer = tf.summary.FileWriter('./graphs2', tf.get_default_graph())
