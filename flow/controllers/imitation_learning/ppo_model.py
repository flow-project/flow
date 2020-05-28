import numpy as np
import json
import h5py
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf


class PPONetwork(TFModelV2):
    """
    Custom RLLib PPOModel (using tensorflow keras) to load weights from a pretained policy model (e.g. from imitation learning) and start RL training with loaded weights.
    Subclass of TFModelV2
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        super(PPONetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        h5_path = model_config.get("custom_options").get("h5_load_path", "")

        # setup model with weights loaded in from model in h5 path
        self.setup_model(obs_space, action_space, model_config, num_outputs, h5_path)

        # register variables for base model
        self.register_variables(self.base_model.variables)


    def setup_model(self, obs_space, action_space, model_config, num_outputs, imitation_h5_path):
        """
        Loads/builds model for both policy and value function
        Args:
             obs_space: observation space of env
             action_space: action space of env
             model_config: configuration parameters for model
             num_outputs: number of outputs expected for policy
             imitation_h5_path: path to h5 file containing weights of a pretrained network (empty string if no such file)
        """

        if imitation_h5_path:
            # set base model to be loaded model
            self.base_model = tf.keras.models.load_model(imitation_h5_path)

        else:
            activation = model_config.get("fcnet_activation")
            hiddens = model_config.get("fcnet_hiddens", [])
            vf_share_layers = model_config.get("vf_share_layers")

            # set up model
            inp_layer = tf.keras.layers.Input(shape=obs_space.shape, name="input_layer")
            curr_layer = inp_layer

            # hidden layers and output for policy
            i = 1
            for size in hiddens:
                curr_layer = tf.keras.layers.Dense(size, name="policy_hidden_layer_{}".format(i),
                                                   activation=activation)(curr_layer)
                i += 1

            output_layer_policy = tf.keras.layers.Dense(num_outputs, name="policy_output_layer", activation=None)(
                curr_layer)

            # set up value function
            if not vf_share_layers:
                curr_layer = inp_layer
                i = 1
                for size in hiddens:
                    curr_layer = tf.keras.layers.Dense(size, name="vf_hidden_layer_{}".format(i),
                                                       activation=activation)(curr_layer)
                    i += 1

            output_layer_vf = tf.keras.layers.Dense(1, name="vf_output_layer", activation=None)(curr_layer)

            # build model from layers
            self.base_model = tf.keras.Model(inp_layer, [output_layer_policy, output_layer_vf])



    def forward(self, input_dict, state, seq_lens):
        """
            Overrides parent class's method. Used to pass a input through model and get policy/vf output.
        """

        policy_out, value_out = self.base_model(input_dict["obs_flat"])
        self.value_out = value_out
        return policy_out, state

    def value_function(self):
        """
            Overrides parent class's method. Get value function method.
        """
        return tf.reshape(self.value_out, [-1])

    def import_from_h5(self, import_file):
        self.setup_model(self, self.obs_space, self.action_space, self.model_config, self.num_outputs, import_file)
