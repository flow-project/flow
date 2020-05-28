import numpy as np
import tensorflow as tf


""" Class agnostic helper functions related to tensorflow"""

def build_neural_net(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedfoward neural network for action prediction

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of pass through Neural Network
    """
    output_placeholder = input_placeholder
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for _ in range(n_layers):
            output_placeholder = tf.layers.dense(output_placeholder, size, activation=activation)
        output_placeholder = tf.layers.dense(output_placeholder, output_size, activation=output_activation,name='Output_Layer')
    return output_placeholder

def create_tf_session():
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
    sess = tf.compat.v1.Session(config=config)
    return sess
