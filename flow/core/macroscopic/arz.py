"""Contains the Aw–Rascle–Zhang traffic flow model class.

Aw, A. A. T. M., and Michel Rascle. "Resurrection of 'second order' models of
traffic flow." SIAM journal on applied mathematics 60.3 (2000): 916-938.
"""
from gym.spaces import Box
import numpy as np
from flow.core.macroscopic.base_model import MacroModelEnv
from flow.core.macroscopic.utils import DictDescriptor


PARAMS = DictDescriptor(
    # ======================================================================= #
    #                           Network parameters                            #
    # ======================================================================= #

    ("length", 10000, float, "length of the stretch of highway"),

    ("dx", 100, float, "length of individual sections on the highway. Speeds "
                       "and densities are computed on these sections. Must be "
                       "a factor of the length"),

    ("rho_max", 0.2, float, "maximum density term in the LWR model (in "
                            "veh/m)"),

    ("rho_max_max", 0.2, float, "maximum possible density of the network (in "
                                "veh/m)"),

    ("v_max", 27.5, float, "initial speed limit of the LWR model. If not "
                           "actions are provided during the simulation "
                           "procedure, this value is kept constant throughout "
                           "the simulation."),

    ("v_max_max", 27.5, float, "max speed limit that the network can be "
                               "assigned"),

    ("CFL", 0.95, float, "Courant-Friedrichs-Lewy (CFL) condition. Must be a "
                         "value between 0 and 1."),

    ("tau", 0.1, float, "time needed to adjust the velocity of a vehicle from "
                        "its current value to the equilibrium speed (in sec)"),

    # ======================================================================= #
    #                          Simulation parameters                          #
    # ======================================================================= #

    ("total_time", 500, float, "time horizon (in seconds)"),

    ("dt", 1, float, "time discretization (in seconds/step)"),

    # ======================================================================= #
    #                      Initial / boundary conditions                      #
    # ======================================================================= #

    ("initial_conditions", ([0 for _ in range(100)], [0 for _ in range(100)]),
     None,  # FIXME
     "tuple of (density, speed) initial conditions. Each element of the tuple"
     " must be a list of length int(length/dx)"),

    ("boundary_conditions", (0, 0), None, "TODO: define what that is"),
)


class ARZ(MacroModelEnv):
    """Aw–Rascle–Zhang traffic flow model.

    Aw, A. A. T. M., and Michel Rascle. "Resurrection of 'second order' models
    of traffic flow." SIAM journal on applied mathematics 60.3 (2000): 916-938.

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

    Attributes
    ----------
    params : dict
        environment-specific parameters. See PARAMS object for more
    length : float
        length of the stretch of highway
    dx : float
        length of individual sections on the highway. Speeds and densities are
        computed on these sections. Must be a factor of the length
    rho_max : float
        maximum density term in the LWR model (in veh/m)
    rho_max_max : float
        maximum possible density of the network (in veh/m)
    v_max : float
        initial speed limit of the LWR model. If not actions are provided
        during the simulation procedure, this value is kept constant throughout
        the simulation.
    v_max_max : float
        max speed limit that the network can be assigned
    CFL : float
        Courant-Friedrichs-Lewy (CFL) condition. Must be a value between 0-1.
    tau : float
        time needed to adjust the velocity of a vehicle from its current value
        to the equilibrium speed (in sec)
    total_time : float
        time horizon (in seconds)
    dt : float
        time discretization (in seconds/step)
    horizon : int
        environment's time horizon, in steps
    initial_conditions : (array_like, array_like)
        tuple of (density, speed) initial conditions. Each element of the tuple
        must be a list of length int(length/dx)
    boundary_left : float
        left boundary conditions
    boundary_right : float
        right boundary conditions
    obs : array_like
        current observation
    num_steps : int
        number of simulation steps since the start of the most recent rollout
    """

    def __init__(self, params):
        """Instantiate the ARZ model.

        Parameters
        ----------
        params : dict
            environment-specific features. Contains the following elements:

            * length (float): length of the stretch of highway
            * dx (float): length of individual sections on the highway. Speeds
              and densities are computed on these sections. Must be a factor of
              the length
            * rho_max (float): maximum density term in the LWR model (in veh/m)
            * rho_max_max (float): maximum possible density of the network (in
              veh/m)
            * v_max (float): initial speed limit of the LWR model. If not
              actions are provided during the simulation procedure, this value
              is kept constant throughout the simulation.
            * v_max_max (float): max speed limit that the network can be
              assigned
            * CFL (float): Courant-Friedrichs-Lewy (CFL) condition. Must be a
              value between 0 and 1.
            * tau (float): time needed to adjust the velocity of a vehicle from
              its current value to the equilibrium speed (in sec)
            * total_time (float): time horizon (in seconds)
            * dt (float): time discretization (in seconds/step)
            * initial_conditions ((list of float, list of float)): tuple of
              (speed, density) initial conditions. Each element of the tuple
              must be a list of length int(length/dx)
            * boundary_conditions ((float, float)): TODO: define what that is
        """
        super(ARZ, self).__init__(params)

        assert (params['length'] / params['dx']).is_integer(), \
            "The 'length' variable in params must be divisible by 'dx'."

        assert (params['total_time'] / params['dt']).is_integer(), \
            "The 'total_time' variable in params must be divisible by 'dt'."

        assert params['rho_max'] <= params['rho_max_max'], \
            "The 'rho_max' must be less than or equal to 'rho_max_max'"

        assert params['v_max'] <= params['v_max_max'], \
            "The 'v_max' must be less than or equal to 'v_max_max'"

        assert 0 <= params['CFL'] <= 1, \
            "'CFL' variable must be between 0 and 1"

        assert params['dt'] <= params['CFL'] * params['dx']/params['v_max'], \
            "CFL condition not satisfied. Make sure dt <= CFL * dx / v_max"

        assert len(params['initial_conditions'][0]) == \
            int(params['length'] / params['dx']), \
            "Initial conditions must be a list of size: length/dx."

        assert len(params['initial_conditions'][1]) == \
            int(params['length'] / params['dx']), \
            "Initial conditions must be a list of size: length/dx."

        assert (params['length'] / params['dx']).is_integer(), \
            "The 'length' variable in params must be divisible by 'dx'."

        self.params = params.copy()
        self.length = params['length']
        self.dx = params['dx']
        self.rho_max = params['rho_max']
        self.rho_max_max = params['rho_max_max']
        self.v_max = params['v_max']
        self.v_max_max = params['v_max_max']
        self.CFL = params['CFL']
        self.tau = params['tau']
        self.total_time = params['total_time']
        self.dt = params['dt']
        self.horizon = int(self.total_time / self.dt)
        self.initial_conditions = params['initial_conditions']
        self.boundary_left = params['boundary_conditions'][0]
        self.boundary_right = params['boundary_conditions'][1]

        self.obs = None
        self.num_steps = None

    @property
    def observation_space(self):
        """See parent class."""
        n_sections = int(self.params['length'] / self.params['dx'])
        return Box(low=0, high=1, shape=(2 * n_sections,), dtype=np.float32)

    @property
    def action_space(self):
        """See parent class."""
        v_max_max = self.params['v_max_max']
        return Box(low=0, high=v_max_max, shape=(1,), dtype=np.float32)

    def step(self, action):
        """TODO."""
        # TODO: fill in
        # Note: if action is set to None, v_max should simply not change from
        # it's last value
        raise NotImplementedError

    def reset(self):
        """Reset the environment.

        Returns
        -------
        array_like
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        # reset the step counter
        self.num_steps = 0

        # reset max speed of the links in the network
        self.v_max = self.params['v_max']

        # reset the observation to match the initial condition
        initial_conditions = np.asarray(self.initial_conditions)
        rho_init = np.asarray(initial_conditions[0]) / self.rho_max_max
        v_init = np.asarray(initial_conditions[1]) / self.v_max_max
        self.obs = np.concatenate((rho_init, v_init))

        return self.obs.copy()
