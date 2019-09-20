"""Script containing the base macro-model object.

This class provides an environmental configuration for generating trainable /
simulate-able models of traffic at a macroscopic scale.

See flow/core/models/to_micro.py for porting these models to integrating macro-
and microscopic models.
"""
import gym


class MacroModel(gym.Env):
    """Base macro-model object.

    This class provides an environmental configuration for generating
    trainable / simulate-able models of traffic at a macroscopic scale.

    See flow/core/models/to_micro.py for porting these models to integrating
    macro- and microscopic models.

    Attributes
    ----------
    net_params : dict
        network-specific features. See the definition of the separate models
        for more.
    """

    def __init__(self, net_params):
        """Instantiate the macro-model object.

        Parameters
        ----------
        net_params : dict
            network-specific features. See the definition of the separate
            models for more.
        """
        self.net_params = net_params

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
