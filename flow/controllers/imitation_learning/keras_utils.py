import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

def build_neural_net_deterministic(input_dim, action_dim, fcnet_hiddens):
    """Build a keras model to output a deterministic policy.
    Parameters
    ----------
    input_dim : int
        dimension of input layer
    action_dim : int
        action_space dimension
    fcnet_hiddens : list
        list containing size of each hidden layer (length of list is number of hidden layers)

    Returns
    -------
    Keras model (untrained)
    """

    input_layer = Input(shape=(input_dim, ))
    curr_layer = input_layer

    for i in range(len(fcnet_hiddens)):
        size = fcnet_hiddens[i]
        dense = Dense(size, activation="tanh")
        curr_layer = dense(curr_layer)
    output_layer = Dense(action_dim, activation=None)(curr_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="policy_network")

    return model

def build_neural_net_stochastic(input_dim, action_dim, fcnet_hiddens):
    """Build a keras model to output a stochastic policy.
    Parameters
    ----------
    input_dim : int
        dimension of input layer
    action_dim : int
        action_space dimension
    fcnet_hiddens : list
        list containing size of each hidden layer (length of list is number of hidden layers)

    Returns
    -------
    Keras model (untrained)
    """
    input_layer = Input(shape=(input_dim, ))
    curr_layer = input_layer

    for i in range(len(fcnet_hiddens)):
        size = fcnet_hiddens[i]
        dense = Dense(size, activation="tanh")
        curr_layer = dense(curr_layer)

    out = Dense(2 * action_dim, activation=None)(curr_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=out, name="policy_network")

    return model

def get_loss(stochastic, variance_regularizer):
    """Get appropriate loss function for training.
    Parameters
    ----------
    stochastic : bool
        determines if policy to be learned is deterministic or stochastic
    variance_regularizer : float
        regularization hyperparameter to penalize high variance policies

    Returns
    -------
    Keras loss function to use for imitation learning.
    """
    if stochastic:
        return negative_log_likelihood_loss(variance_regularizer)
    else:
        return tf.keras.losses.mean_squared_error

def negative_log_likelihood_loss(variance_regularizer):
    """Negative log likelihood loss for learning stochastic policies.

    Parameters
    ----------
    variance_regularizer : float
        regularization hyperparameter to penalize high variance policies
    Returns
    -------
    Negative log likelihood loss function with variance regularization.
    """

    def nll_loss(y, network_output):
        assert network_output.shape[1] % 2 == 0, "Stochastic policies must output vectors of even length"

        action_dim = network_output.shape[1] // 2

        # first half of network_output is mean, second half is log_std
        means, log_stds = network_output[:, :action_dim], network_output[:, action_dim:]
        stds = tf.math.exp(log_stds)
        variances = tf.math.square(stds)

        # Multivariate Gaussian distribution
        dist = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=variances)
        loss = dist.log_prob(y)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss) + (variance_regularizer * tf.norm(variances))
        return loss

    return nll_loss
