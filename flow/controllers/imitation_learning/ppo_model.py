import numpy as np

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import get_activation_fn, try_import_tf

tf = try_import_tf()



class PPONetwork(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        super(PPONetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.setup_model(obs_space, action_space, model_config, num_outputs, None)

    def setup_model(self, obs_space, action_space, model_config, num_outputs, load_path):
        if load_path:
            try:
                loaded_policy_model = tf.keras.load_model(load_path)
                inp_layer = loaded_policy_model.input
                curr_layer = loaded_policy_model.layers[-2].output

            except Exception as e:
                print("Error in loading existing model specified by load_path")
                raise e
        else:
            activation = get_activation_fn(model_config.get("fcnet_activation"))
            hiddens = model_config.get("fcnet_hiddens", [])
            vf_share_layers = model_config.get("vf_share_layers")

            inp_layer = tf.keras.layers.Input(shape=obs_space.shape, name="input_layer")
            curr_layer = inp_layer

            i = 1
            for size in hiddens:
                curr_layer = tf.keras.layers.Dense(size, name="policy_hidden_layer_{}".format(i), activation=activation)(curr_layer)
                i += 1

            output_layer_policy = tf.keras.layers.Dense(num_outputs, name="policy_output_layer", activation=None)(curr_layer)

        if not vf_share_layers:
            curr_layer = inp_layer
            i = 1
            for size in hiddens:
                curr_layer = tf.keras.layers.Dense(size, name="vf_hidden_layer_{}".format(i), activation=activation)(curr_layer)
                i += 1

        output_layer_vf = tf.keras.layers.Dense(1, name="vf_output_layer", activation=None)(curr_layer)

        self.base_model = tf.keras.Model(inp_layer, [output_layer_policy, output_layer_vf])
        self.register_variables(self.base_model.variables)


    def forward(self, input_dict, state, seq_lens):
        policy_out, value_out = self.base_model(input_dict["obs_flat"])
        self.value_out = value_out
        return policy_out, state

    def value_function(self):
        return tf.reshape(self.value_out, [-1])

    def import_from_h5(self, import_file):
        self.setup_model(self, self.obs_space, self.action_space, self.model_config, self.num_outputs, import_file)




