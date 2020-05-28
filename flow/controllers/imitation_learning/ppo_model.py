import numpy as np
import json
import h5py
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf


class PPONetwork(TFModelV2):
    """
    Custom RLLib PPOModel (using tensorflow keras) to load weights from a pre-trained policy model (e.g. from imitation learning) and start RL training with loaded weights.
    Subclass of TFModelV2. See https://docs.ray.io/en/master/rllib-models.html.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        Parameters
        __________
        obs_space: gym.Space
            observation space of gym environment
        action_space: gym.Space
            action_space of gym environment
        num_outputs: int
            number of outputs for policy network. For deterministic policies, this is dimension of the action space. For continuous stochastic policies, this is 2 * dimension of the action space
        model_config: dict
            configuration of model
        name: str
            name of model

        """

        super(PPONetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        h5_path = model_config.get("custom_options").get("h5_load_path", "")

        # setup model with weights loaded in from model in h5 path
        self.setup_model(obs_space, action_space, model_config, num_outputs, h5_path)

        # register variables for base model
        self.register_variables(self.base_model.variables)


    def setup_model(self, obs_space, action_space, model_config, num_outputs, imitation_h5_path):
        """
        Loads/builds model for both policy and value function
        Parameters
        __________

        obs_space: gym.Space
            observation space of env
        action_space: gym.Space
            action space of env
        model_config: dict
            configuration parameters for model
        num_outputs: int
            number of outputs expected for policy
        imitation_h5_path: str
            path to h5 file containing weights of a pretrained network (empty string if no such file)
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
        Parameters
        __________
        input_dict: dict
            dictionary of input tensors, including “obs”, “obs_flat”, “prev_action”, “prev_reward”, “is_training”
        state: list
            list of state tensors with sizes matching those returned by get_initial_state + the batch dimension
        seq_lens: tensor
            1d tensor holding input sequence lengths

        Returns
        _______
        (outputs, state)
            Tuple, first element is policy output, second element state
        """

        policy_out, value_out = self.base_model(input_dict["obs_flat"])
        self.value_out = value_out
        return policy_out, state

    def value_function(self):
        """
        Returns the value function output for the most recent forward pass.

        Returns
        _______
        tensor
            value estimate tensor of shape [BATCH].
        """
        return tf.reshape(self.value_out, [-1])

    def import_from_h5(self, import_file):
        """
        Overrides parent class method. Import base_model from h5 import_file.
        Parameters:
         __________
        import_file: str
            filepath to h5 file
        """
        self.setup_model(self, self.obs_space, self.action_space, self.model_config, self.num_outputs, import_file)
