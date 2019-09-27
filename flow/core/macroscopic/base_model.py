"""Script containing the base macroscopic traffic environment object.

This class provides an environmental configuration for generating trainable /
simulate-able models of traffic at a macroscopic scale.

See flow/core/models/to_micro.py for porting these models to integrating macro-
and microscopic models.
"""
import gym


class MacroModelEnv(gym.Env):
    """Base macroscopic traffic environment object.

    This class provides an environmental configuration for generating
    trainable / simulate-able models of traffic at a macroscopic scale.

    See flow/core/models/to_micro.py for porting these models to integrating
    macro- and microscopic models.

    Attributes
    ----------
    params : dict
        environment-specific features. See the definition of the separate
        models for more.
    """

    def __init__(self, params):
        """Instantiate the macro-model object.

        Parameters
        ----------
        params : dict
            environment-specific features. See the definition of the separate
            models for more.
        """
        self.params = params

    def step(self, action):
        """Advance the simulation by one step.

        Parameters
        ----------
        action : Any, optional
            specifies that action to be performed by trainable agents in the
            environment. If set to None, no new action is performed.

        Returns
        -------
        array_like
            agent's observation of the current environment
        float
            amount of reward associated with the previous state/action pair
        bool
            indicates whether the episode has ended
        dict
            contains other diagnostic information from the previous action
        """
        raise NotImplementedError

    def reset(self):
        """Reset the simulation.

        Returns
        -------
        array_like
            the initial observation of the space
        """
        raise NotImplementedError

    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box or Tuple
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box or Tuple
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """See parent class."""
        pass
