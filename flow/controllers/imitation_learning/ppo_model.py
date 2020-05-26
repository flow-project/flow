import numpy as np
import json
import h5py
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import get_activation_fn, try_import_tf
# from flow.controllers.imitation_learning.keras_utils import *

tf = try_import_tf()



class PPONetwork(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        super(PPONetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # setup model with weights loaded in from model in h5 path
        self.setup_model(obs_space, action_space, model_config, num_outputs, '/Users/akashvelu/Desktop/follower_stopper1.h5')
        # register variables for base model
        self.register_variables(self.base_model.variables)


    def setup_model(self, obs_space, action_space, model_config, num_outputs, imitation_h5_path):

        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens", [])
        vf_share_layers = model_config.get("vf_share_layers")

        # set up model
        inp_layer = tf.keras.layers.Input(shape=obs_space.shape, name="input_layer")
        curr_layer = inp_layer

        # hidden layers and output for policy
        i = 1
        for size in hiddens:
            curr_layer = tf.keras.layers.Dense(size, name="policy_hidden_layer_{}".format(i), activation=activation)(curr_layer)
            i += 1

        output_layer_policy = tf.keras.layers.Dense(num_outputs, name="policy_output_layer", activation=None)(curr_layer)

        # set up value function
        if not vf_share_layers:
            curr_layer = inp_layer
            i = 1
            for size in hiddens:
                curr_layer = tf.keras.layers.Dense(size, name="vf_hidden_layer_{}".format(i), activation=activation)(curr_layer)
                i += 1

        output_layer_vf = tf.keras.layers.Dense(1, name="vf_output_layer", activation=None)(curr_layer)

        # build model from layers
        self.base_model = tf.keras.Model(inp_layer, [output_layer_policy, output_layer_vf])


        if imitation_h5_path:
            # imitation_model = tf.keras.models.load_model(imitation_h5_path, custom_objects={"negative_log_likelihood_loss": negative_log_likelihood_loss})

            # set up a model to load in weights from imitation network (without the training variables, e.g. adam variables)
            imitation_inp = tf.keras.layers.Input(shape=obs_space.shape, name="imitation_inp")
            curr_imitation_layer = imitation_inp
            i = 1
            for size in hiddens:
                curr_imitation_layer = tf.keras.layers.Dense(size, name="imitation_hidden_layer_{}".format(i), activation=activation)(curr_imitation_layer)
                i += 1

            imitation_output_layer = tf.keras.layers.Dense(num_outputs, name="imitation_output_layer", activation=None)(curr_imitation_layer)
            imitation_model = tf.keras.Model(imitation_inp, [imitation_output_layer])

            # load weights from file into model
            imitation_model.load_weights(imitation_h5_path)
            # register model variables (to prevent error)
            self.register_variables(imitation_model.variables)

            # copy these weights into the base model (only the policy hidden layer and output weights)
            for i in range(len(hiddens)):
                imitation_layer = imitation_model.layers[i + 1]
                base_model_layer_name = 'policy_hidden_layer_' + str(i + 1)
                base_model_layer = self.base_model.get_layer(base_model_layer_name)
                base_model_layer.set_weights(imitation_layer.get_weights())

            imitation_layer = imitation_model.layers[-1]
            base_model_layer_name = 'policy_output_layer'
            base_model_layer = self.base_model.get_layer(base_model_layer_name)
            base_model_layer.set_weights(imitation_layer.get_weights())


    def forward(self, input_dict, state, seq_lens):
        policy_out, value_out = self.base_model(input_dict["obs_flat"])
        self.value_out = value_out
        return policy_out, state

    def value_function(self):
        return tf.reshape(self.value_out, [-1])

    def import_from_h5(self, import_file):
        self.setup_model(self, self.obs_space, self.action_space, self.model_config, self.num_outputs, import_file)
