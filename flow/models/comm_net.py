from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.models.model import Model
from ray.rllib.models.misc import normc_initializer, get_activation_fn

### Added by Lucas Fischer on 10/24/18
import numpy as np

class Commnet(Model):
    """Generic fully connected network."""

    # def _build_layers(self, inputs, num_outputs, options):
    ### Added by Lucas Fischer on 10/24/18
    # hiddens = options.get("fcnet_hiddens", [256, 256])
    # activation = get_activation_fn(options.get("fcnet_activation", "tanh"))
    # with tf.name_scope("fc_net"):
    #    i = 1
    #    last_layer = inputs
    #    for size in hiddens:
    #        label = "fc{}".format(i)
    #        last_layer = slim.fully_connected(
    #            last_layer,
    #            size,
    #            weights_initializer=normc_initializer(1.0),
    #            activation_fn=activation,
    #            scope=label)
    #        i += 1
    #    label = "fc_out"
    #    output = slim.fully_connected(
    #        last_layer,
    #        num_outputs,
    #        weights_initializer=normc_initializer(0.01),
    #        activation_fn=None,
    #        scope=label)
    #    return output, last_layer
    @staticmethod
    def _build_layers(self, inputs, num_outputs, options):
        custom_name = options["custom_options"].get("name", "test_name")
        hidden_vector_len = options["custom_options"].get("hidden_vector_len", 20)
        # remove all zero rows from inputs
        intermediate_tensor = tf.reduce_sum(tf.abs(inputs), 1)
        zero_vector = tf.zeros(shape=(1, 1), dtype=tf.float32)
        bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
        omit_zeros = tf.boolean_mask(inputs, bool_mask)
        inputs = inputs[omit_zeros]
        with tf.variable_scope(custom_name):
            H = self.base_build_network(inputs, num_outputs, hidden_vector_len)
            return self.actor_output_layer(H, num_outputs, hidden_vector_len)

    @staticmethod
    def base_build_network(self, observation, num_outputs, hidden_len):

        H0 = observation
        # vector of communication
        C0 = tf.zeros(tf.shape(H0), name="C0")
        H1, C1 = self.comm_step("comm_step1", H0, C0, hidden_len)
        H2, _ = self.comm_step("comm_step2", H1, C1, hidden_len, H0)

        return H2

    @staticmethod
    def comm_step(self, name, H, C, hidden_vector_len, H0_skip_con=None):
        batch_size = tf.shape(H)[0]
        with tf.variable_scope(name):
            next_H = tf.zeros(shape=(batch_size, 0, hidden_vector_len))
            for j in range(NUM_AGENTS):
                h = H[:, j]
                c = C[:, j]

                next_h = self.module(h, c, hidden_vector_len)  # shape (BATCH_SIZE, HIDDEN_VECTOR_LEN)
                next_H = tf.concat([next_H, tf.reshape(next_h, (batch_size, 1, hidden_vector_len))], 1)

            next_H = tf.identity(next_H, "H")

            if H0_skip_con is not None:
                next_H = tf.add(next_H, H0_skip_con)

            if NUM_AGENTS > 1:
                next_C = tf.zeros(shape=(batch_size, 0, hidden_vector_len))
                for j1 in range(NUM_AGENTS):
                    next_c = []
                    for j2 in range(NUM_AGENTS):
                        if j1 != j2:
                            next_c.append(next_H[:, j2])
                    next_c = tf.reduce_mean(tf.stack(next_c), 0)
                    next_C = tf.concat([next_C, tf.reshape(next_c, (batch_size, 1, hidden_vector_len))], 1)
            else:
                next_C = C

            return next_H, tf.identity(next_C, "C")

    @staticmethod
    def module(self, h, c, hidden_vector_len):
        with tf.variable_scope("module", reuse=tf.AUTO_REUSE):
            w_H = tf.get_variable(name='w_H', shape=hidden_vector_len,
                                  initializer=tf.contrib.layers.xavier_initializer())
            w_C = tf.get_variable(name='w_C', shape=hidden_vector_len,
                                  initializer=tf.contrib.layers.xavier_initializer())

            return tf.tanh(tf.multiply(w_H, h) + tf.multiply(w_C, c))

    @staticmethod
    def actor_output_layer(self, H, num_outputs, hidden_vector_len):
        with tf.variable_scope("actor_output"):
            w_out = tf.get_variable(name='w_out', shape=(hidden_vector_len, num_outputs),
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_out = tf.get_variable(name='b_out', shape=num_outputs, initializer=tf.zeros_initializer())

            batch_size = tf.shape(H)[0]

            actions = []
            for j in range(NUM_AGENTS):
                h = tf.slice(H, [0, j, 0], [batch_size, 1, hidden_vector_len])
                w_out_batch = tf.tile(tf.expand_dims(w_out, axis=0), [batch_size, 1, 1])
                action = tf.squeeze(tf.matmul(h, w_out_batch) + b_out, [1])

                actions.append(action)
            actions = tf.stack(actions, name="actions", axis=1)

        return actions