import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

def build_neural_net_deterministic(input_dim, action_dim, n_layers, size):
    input_layer = Input(shape=(input_dim, ))
    curr_layer = input_layer

    for _ in range(n_layers):
        dense = Dense(size, activation="tanh")
        curr_layer = dense(curr_layer)
    output_layer = Dense(action_dim, activation=None)(curr_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="policy_network")

    return model

def build_neural_net_stochastic(input_dim, action_dim, n_layers, size):
    input_layer = Input(shape=(input_dim, ))
    curr_layer = input_layer

    for _ in range(n_layers):
        dense = Dense(size, activation="tanh")
        curr_layer = dense(curr_layer)

    out = Dense(2 * action_dim, activation=None)(curr_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=out, name="policy_network")

    return model

def get_loss(stochastic):
    if stochastic:
        return negative_log_likelihood_loss
    else:
        return tf.keras.losses.mean_squared_error

def negative_log_likelihood_loss(y, distribution_params):
    assert distribution_params.shape[1] % 2 == 0, "Stochastic policies must output vectors of even length"
    action_dim = distribution_params.shape[1]//2
    means, log_stds = distribution_params[:, :action_dim], distribution_params[:, action_dim:]
    stds = tf.math.exp(log_stds)
    variances = tf.math.square(stds)
    dist = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=variances)
    loss = dist.log_prob(y)
    loss = tf.negative(loss)
    loss = tf.reduce_mean(loss) + (0.5 * tf.norm(variances))
    return loss
