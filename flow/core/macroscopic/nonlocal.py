"""

"""
from flow.core.macroscopic.base_model import MacroModel

NET_PARAMS = {
    # length of the stretch of highway
    "length": 10000,
    # length of individual sections on the highway. Speeds and densities are
    # computed on these sections. Must be a factor of the length
    "dx": 100,
    # tuple of (speed, density) initial conditions. Each element of the tuple
    # must be a list of length int(length/dx)
    "initial_conditions": [0 for _ in range(100)],
    # boundary conditions  TODO: define what that is
    "boundary_conditions": None,  # FIXME
    # TODO: add model parameters
}


class NonLocalModel(MacroModel):
    """

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
        # TODO: fill in
        # Note: if action is set to None, v_max should simply not change from
        # it's last value
        raise NotImplementedError

    def reset(self):
        # TODO: fill in
        raise NotImplementedError
