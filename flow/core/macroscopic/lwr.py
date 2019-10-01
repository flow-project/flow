"""Contains the Lighthill-Whitham-Richards traffic flow model class.

M.J.Lighthill, G.B.Whitham, On kinematic waves II: A theory of traffic flow on
long, crowded roads. Proceedings of the Royal Society of London Series A 229,
317-345, 1955
"""
from gym.spaces import Box
import numpy as np
from flow.core.macroscopic.base_model import MacroModelEnv
from flow.core.macroscopic.utils import DictDescriptor


PARAMS = DictDescriptor(
    # ======================================================================= #
    #                           Network parameters                            #
    # ======================================================================= #

    ("length", 35, float, "length of the stretch of highway"),

    ("dx", 35/150, float, "length of individual sections on the highway. "
                          "Speeds and densities are computed on these "
                          "sections. Must be a factor of the length"),

    ("rho_max", 4, float, "maximum density term in the LWR model (in veh/m)"),

    ("rho_max_max", 4, float, "maximum possible density of the network (in "
                              "veh/m)"),

    ("v_max", 1, float, "initial speed limit of the LWR model. If not actions "
                        "are provided during the simulation procedure, this "
                        "value is kept constant throughout the simulation."),

    ("v_max_max", 1, float, "max speed limit that the network can be "
                            "assigned"),

    ("CFL", 0.95, float, "Courant-Friedrichs-Lewy (CFL) condition. Must be a "
                         "value between 0 and 1."),

    # ======================================================================= #
    #                          Simulation parameters                          #
    # ======================================================================= #

    ("total_time", 110.5, float, "time horizon (in seconds)"),

    ("dt", 0.221, float, "time discretization (in seconds/step)"),

    # ======================================================================= #
    #                      Initial / boundary conditions                      #
    # ======================================================================= #

    ("initial_conditions", [0], None,  # FIXME
     "list of initial densities. Must be of length int(length/dx)"),

    ("boundary_conditions", (None, None), None, "TODO: define what that is"),
)


class LWR(MacroModelEnv):
    """Lighthill-Whitham-Richards traffic flow model.

    M.J.Lighthill, G.B.Whitham, On kinematic waves II: A theory of traffic flow
    on long, crowded roads. Proceedings of the Royal Society of London Series A
    229, 317-345, 1955

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
    total_time : float
        time horizon (in seconds)
    dt : float
        time discretization (in seconds/step)
    horizon : int
        environment's time horizon, in steps
    initial_conditions : array_like
        list of initial densities. Must be of length int(length/dx)
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
        """Instantiate the LWR model.

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
            * total_time (float): time horizon (in seconds)
            * dt (float): time discretization (in seconds/step)
            * initial_conditions (list of float): list of initial densities.
              Must be of length int(length/dx)
            * boundary_conditions ((float, float)): TODO: define what that is
        """
        super(LWR, self).__init__(params)

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

        assert len(params['initial_conditions']) == \
            int(params['length'] / params['dx']), \
            "Initial conditions must be a list of size: length/dx."

        self.params = params.copy()
        self.length = params['length']
        self.dx = params['dx']
        self.rho_max = params['rho_max']
        self.rho_max_max = params['rho_max_max']
        self.v_max = params['v_max']
        self.v_max_max = params['v_max_max']
        self.CFL = params['CFL']
        self.total_time = params['total_time']
        self.dt = params['dt']
        self.horizon = int(self.total_time / self.dt)
        self.initial_conditions = params['initial_conditions']
        self.boundary_left = params['boundary_conditions'][0]
        self.boundary_right = params['boundary_conditions'][1]
        self.speeds = None

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

    def step(self, rl_actions):
        """Advance the simulation by a single step.

        Parameters
        ----------
        rl_actions : int or array_like
            actions to be performed by the agent

        Returns
        -------
        array_like
            next observation
        float
            reward
        bool
            done mask
        dict
            additional information (defaults to {})
        """
        # increment the step counter
        self.num_steps += 1

        # update the v_max value is an action is provided
        if rl_actions is not None:
            self.v_max = rl_actions

        # advance the state of the simulation by one step
        rho = self.ibvp(
            self.obs[:int(self.length/self.dx)],
            self.boundary_right,
            self.boundary_left,
        )
        speed = self.speed_info(rho)

        # compute the new observation
        self.obs = np.concatenate((rho / 1, speed / 1))

        # compute the reward value
        rew = np.mean(np.square(speed - self.v_max) * rho)

        # compute the done mask
        done = self.num_steps >= self.horizon

        return self.obs.copy(), rew, done, {}

    def godunov_flux(self, rho_t):
        """Calculate the Godunov numerical flux vector of our data.

        Parameters
        ----------
        rho_t : array_like
           densities containing boundary conditions to be analysed

        Returns
        -------
        array_like
            array of fluxes calibrated at every point of our data
        """
        # demand
        d = self.v_max * rho_t * (1 - rho_t / self.rho_max) \
            * (rho_t < 0.5 * self.rho_max) \
            + 0.25 * self.v_max * self.rho_max * (rho_t >= 0.5 * self.rho_max)

        # supply
        s = self.v_max * rho_t * (1 - rho_t / self.rho_max) \
            * (rho_t > 0.5 * self.rho_max) \
            + 0.25 * self.v_max * self.rho_max * (rho_t <= 0.5 * self.rho_max)
        s = np.append(s[1:], s[len(s) - 1])

        # Godunov flux
        return np.minimum(d, s)

    def ibvp(self, rho_t, u_right, u_left):
        """Implement Godunov scheme for multi-populations.

        Friedrich, Jan & Kolb, Oliver & Goettlich, Simone. (2018). A Godunov
        type scheme for a class of LWR traffic flow models with non-local flux.
        Networks & Heterogeneous Media. 13. 10.3934/nhm.2018024.

        Parameters
        ----------
        rho_t : array_like
            density data to be analyzed and calculate next points for this data
            using Godunov scheme.
            Note: at time = 0, rho_t = initial density data
        u_right : double
            right boundary condition
        u_left : double
            left boundary condition

        Returns
        -------
        array_like
              next density data points as calculated by the Godunov scheme
        """
        # lam = time/distance step
        lam = self.dt / self.dx

        rho_t = np.insert(np.append(rho_t, u_right), 0, u_left)

        # Godunov numerical flux
        f = self.godunov_flux(rho_t)

        fm = np.insert(f[0:len(f) - 1], 0, f[0])

        # Godunov scheme (updating rho_t)
        rho_t = rho_t - lam * (f - fm)

        rho_t = np.insert(np.append(rho_t[1:len(rho_t) - 1], u_right),
                          0, u_left)

        # remove the boundary conditions from the output
        rho_t = rho_t[1:len(rho_t) - 1]

        return rho_t

    def speed_info(self, density):
        """Implement the Greenshields model for the equilibrium velocity.

        Greenshields, B. D., Ws Channing, and Hh Miller. "A study of traffic
        capacity." Highway research board proceedings. Vol. 1935. National
        Research Council (USA), Highway Research Board, 1935.

        Parameters
        ----------
        density : array_like
            densities of every specified point on the road

        Returns
        -------
        array_like
            equilibrium velocity at every specified point on road
        """
        self.speeds = self.v_max * (1 - (density / self.rho_max))
        return self.speeds

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
        rho_init = initial_conditions
        v_init = self.speed_info(initial_conditions)
        self.obs = np.concatenate((rho_init, v_init))

        return self.obs.copy()
