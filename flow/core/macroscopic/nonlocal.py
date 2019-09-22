"""Contains the non-local traffic flow model class.

TODO: add citation
"""
from flow.core.macroscopic.base_model import MacroModelEnv
from flow.core.macroscopic.utils import DictDescriptor


NET_PARAMS = DictDescriptor(

)


class NonLocalModel(MacroModelEnv):
    """Non-local traffic flow model class.

    TODO: add citation

    States
        The observation consists of the normalized densities and speeds of the
        individual nodes in the network.

    Actions
        The actions update the v_max values of the nodes of the network. If set
        to None, the v_max values is not updated.

    Rewards
        The reward function is the average L2 distance between the speeds of
        the individual nodes and the maximum achievable speed, weighted by the
        densities of the individual nodes.

    Termination
        A rollout is terminated if the time horizon.
    """

    def __init__(self, net_params):
        """Instantiate the non-local model.

        :param net_params:
        """
        super(NonLocalModel, self).__init__(net_params)

        assert (net_params['length'] / net_params['dx']).is_integer(), \
            "The 'length' variable in net_params must be divisible by 'dx'."

        # TODO: fill in

    def step(self, action):
        """TODO."""
        # TODO: fill in
        # Note: if action is set to None, v_max should simply not change from
        # it's last value
        raise NotImplementedError

    def reset(self):
        """TODO."""
        # TODO: fill in
        raise NotImplementedError
