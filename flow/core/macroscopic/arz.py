"""Contains the Aw–Rascle–Zhang traffic flow model class.

Aw, A. A. T. M., and Michel Rascle. "Resurrection of 'second order' models of
traffic flow." SIAM journal on applied mathematics 60.3 (2000): 916-938.
"""
from gym.spaces import Box
import numpy as np
from scipy.optimize import fsolve
from flow.core.macroscopic.base_model import MacroModelEnv
from flow.core.macroscopic.utils import DictDescriptor


PARAMS = DictDescriptor(
    # ======================================================================= #
    #                           Network parameters                            #
    # ======================================================================= #

    ("length", 10, float, "length of the stretch of highway"),

    ("dx", 10/150, float, "length of individual sections on the highway. "
                          "Speeds and densities are computed on these "
                          "sections. Must be a factor of the length"),

    ("rho_max", 1, float, "maximum density term in the LWR model (in veh/m)"),

    ("rho_max_max", 1, float, "maximum possible density of the network (in "
                              "veh/m)"),

    ("v_max", 1, float, "initial speed limit of the LWR model. If not actions "
                        "are provided during the simulation procedure, this "
                        "value is kept constant throughout the simulation."),

    ("v_max_max", 1, float, "max speed limit that the network can be "
                            "assigned"),

    ("CFL", 0.99, float, "Courant-Friedrichs-Lewy (CFL) condition. Must be a "
                         "value between 0 and 1."),

    ("tau", 0.1, float, "time needed to adjust the velocity of a vehicle from "
                        "its current value to the equilibrium speed (in sec)"),

    # ======================================================================= #
    #                          Simulation parameters                          #
    # ======================================================================= #

    ("total_time", 660, float, "time horizon (in seconds)"),

    ("dt", 0.066, float, "time discretization (in seconds/step)"),

    # ======================================================================= #
    #                      Initial / boundary conditions                      #
    # ======================================================================= #

    ("initial_conditions", ([0], [0]),
     None,
     "tuple of (density, speed) initial conditions. Each element of the tuple"
     " must be a list of length int(length/dx)"),

    ("boundary_conditions", (None, None), None,
     "tuple of ((density, speed),tuple of (density, speed)). Specify values "
     "for left boundary and right boundary"),
)


def boundary_right_solve(data):
    """Calculate right boundary condition.

    Parameters
    ----------
    data : tuple -> (density, relative flow)
        density: array_like
            density data points on the road length
       relative flow: array_ike
           relative flow data points on the road length

    Returns
    -------
    tuple -> (double, double)
        right boundary conditions for Data -> (density, relative flow)
    """
    return data[0][len(data[1]) - 1], data[1][len(data[1]) - 1]


def boundary_left_solve(data):
    """Calculate left boundary condition.

    Parameters
    ----------
    data : tuple -> (density, relative flow)
        density: array_like
            density data points on the road length
        relative flow: array_ike
           relative flow data points on the road length

    Returns
    -------
    tuple -> (double, double)
        left boundary conditions for Data -> (density, relative flow)
    """
    return data[0][0], data[1][0]


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
    rho_next : array_like
        the next density value
    rhs : TODO
        right hand side constant for the relative flow (used during fsolve)
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
            int(params['length'] / params['dx']) - 1, \
            "Initial conditions must be a list of size: (length/dx) - 1."

        assert len(params['initial_conditions'][1]) == \
            int(params['length'] / params['dx']) - 1, \
            "Initial conditions must be a list of size: length/dx."

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
        self.initial_conditions_density = params['initial_conditions'][0]
        self.initial_conditions_velocity = params['initial_conditions'][1]
        # calculate relative flow (transform velocity (u) to relative flow (y))
        self.initial_conditions_relative_flow = self.relative_flow(
            self.initial_conditions_density, self.initial_conditions_velocity)
        self.initial_conditions = (self.initial_conditions_density,
                                   self.initial_conditions_relative_flow)
        self.list_values = []
        self.boundary_left = params['boundary_conditions'][0]
        self.boundary_right = params['boundary_conditions'][1]

        self.speed = None
        self.obs = None
        self.num_steps = None
        self.rho_next = None
        self.rhs = None

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
        """See parent class."""
        # increment the step counter
        self.num_steps += 1

        # update the v_max value is an action is provided
        if rl_actions is not None:
            self.v_max = rl_actions

        # advance the state of the simulation by one step
        rho, relative_flow = self.arz_solve(self.obs)
        self.speed = self.u(rho, relative_flow)

        # compute the new observation
        self.obs = (rho / self.rho_max_max,
                    self.relative_flow(rho, self.speed / self.v_max))
        # compute the reward value
        rew = np.mean(np.square(self.speed - self.v_max) * rho)

        # compute the done mask
        done = self.num_steps >= self.horizon

        return self.obs, rew, done, {}

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
        self.obs = self.initial_conditions

        return self.obs

    def arz_solve(self, u_full):
        """Implement Godunov Semi-Implicit scheme for multi-populations.

        Fan, Shimao et al. “Comparative model accuracy of a data-fitted
        generalized Aw-Rascle-Zhang model.” NHM 9 (2014): 239-268.

        Parameters
        ----------
        u_full : tuple -> (density, relative flow)
            density: array_like
                density data points on the road length
            relative flow: array_ike
               relative flow data points on the road length
            Note: at time = 0, u_full = initial density data

        Returns
        -------
        array_like
            next density data points as calculated by the Semi-Implicit Godunov
            scheme
        array_ike
            next relative flow data points as calculated by the Semi-Implicit
            Godunov scheme
        """
        # we are extrapolating our boundary conditions
        self.boundary_left = boundary_left_solve(u_full)
        self.boundary_right = boundary_right_solve(u_full)

        # full array with boundary conditions
        u_all = (np.insert(np.append(u_full[0], self.boundary_right[0]),
                           0, self.boundary_left[0]),
                 np.insert(np.append(u_full[1], self.boundary_right[1]),
                           0, self.boundary_left[1]))

        # compute flux
        fp_higher_half, fp_lower_half, fy_higher_half, fy_lower_half, \
            rho_init, y_init = self.compute_flux(u_all)

        # update new points
        new_points = self.arz_update_points(
            fp_higher_half,
            fp_lower_half,
            fy_higher_half,
            fy_lower_half,
            rho_init,
            y_init,
        )

        return new_points

    def function_rho(self, density, y_value):
        """Calculate flux associated with density using density Flux function.

        Fan, Shimao et al. “Comparative model accuracy of a data-fitted
        generalized Aw-Rascle-Zhang model.” NHM 9 (2014): 239-268.

        Parameters
        ----------
        density : array_like
            density data points at every specified point on the road
        y_value : array_like
            relative flow data points at every specified point on the road

        Returns
        -------
        array_like
            density flux values
        """
        # Flux equation for density (rho)
        return y_value + density * self.ve(density)

    def function_y(self, density, y_value):
        """Calculate flux associated with relative flow.

        Fan, Shimao et al. “Comparative model accuracy of a data-fitted
        generalized Aw-Rascle-Zhang model.” NHM 9 (2014): 239-268.

        Parameters
        ----------
        density : array_like
            density data points at every specified point on the road
        y_value : array_like
            relative flow data points at every specified point on the road

        Returns
        -------
        array_like
            relative flow flux values
        """
        return y_value ** 2 / density + y_value * self.ve(density)

    def compute_flux(self, u_all):
        """Implement the 'The Lax-Friedrichs Method' for Flux calculations.

        "Finite-Volume Methods for Hyperbolic Problems", Chapter 4 (p71),
        Randal J. Leveque.

        Parameters
        ----------
        u_all : tuple -> (density, relative flow)
            density : array_like
                density data points on the road length with boundary conditions
            relative flow : array_ike
               relative flow data points on the road length with boundary
               conditions

            Note: at time = 0, U_all = initial density data

        Returns
        -------
        array_like
            density flux at right boundary of each cell
        array_like
            density flux at left boundary of each cell
        array_like
            relative flow flux at right boundary of each cell
        array_like
            relative flow flux at left boundary of each cell
        array_like
            current values for density at each point on the road (midpoint of
            cell)
        array_like
            current values for relative flow at each point on the road
            (midpoint of cell)
        """
        rho_full = u_all[0]
        y_full = u_all[1]

        # left cell boundary data to be considered -> entire row except last
        # two
        rho_l = rho_full[:-2]
        y_l = y_full[:-2]

        # midpoint cell boundary data to be considered -> all expect first and
        # last
        rho_init = rho_full[1:-1]
        y_init = y_full[1:-1]

        # right cell boundary data to be considered -> entire row except first
        # two
        rho_r = rho_full[2:]
        y_r = y_full[2:]

        # left fluxes
        fp_lower_half = 0.5 \
            * (self.function_rho(rho_l, y_l)
               + self.function_rho(rho_init, y_init)) \
            - ((0.5 * self.dt / self.dx) * (rho_init - rho_l))

        fy_lower_half = 0.5 \
            * (self.function_y(rho_l, y_l)
               + self.function_y(rho_init, y_init)) \
            - ((0.5 * self.dt / self.dx) * (y_init - y_l))

        # right fluxes
        fp_higher_half = 0.5 \
            * (self.function_rho(rho_r, y_r)
               + self.function_rho(rho_init, y_init)) \
            - ((0.5 * self.dt / self.dx) * (rho_r - rho_init))

        fy_higher_half = 0.5 \
            * (self.function_y(rho_r, y_r)
               + self.function_y(rho_init, y_init)) \
            - ((0.5 * self.dt / self.dx) * (y_r - y_init))

        self.list_values = self.list_values + [
            [fp_higher_half, fp_lower_half, fy_higher_half, fy_lower_half,
             rho_init, y_init]
        ]

        return fp_higher_half, fp_lower_half, fy_higher_half, fy_lower_half, \
            rho_init, y_init

    def arz_update_points(self,
                          fp_higher_half,
                          fp_lower_half,
                          fy_higher_half,
                          fy_lower_half,
                          rho_init,
                          y_init):
        """Update our current density and relative flow values.

        Parameters
        ----------
        fp_higher_half : array_like
            density flux at right boundary of each cell
        fp_lower_half : array_like
            density flux at left boundary of each cell
        fy_higher_half : array_like
            relative flow flux at right boundary of each cell
        fy_lower_half : array_like
            relative flow flux at left boundary of each cell
        rho_init : array_like
            current values for density at each point on the road (midpoint of
            cell)
        y_init : array_like
            current values for relative flow at each point on the road
            (midpoint of cell)

        Returns
        -------
        array_like
            next density values at each point on the road
        array_like
            next relative flow values at each point on the road
        """
        # time and cell step
        step = self.dt / self.dx  # where are we referencing this?

        # updating density
        self.rho_next = rho_init + (step * (fp_lower_half - fp_higher_half))

        # updating relative flow
        # right hand side constant -> we use fsolve to find our roots
        self.rhs = y_init + (step * (fy_lower_half - fy_higher_half)) \
            + (self.dt / self.tau) * self.rho_next * self.ve(self.rho_next)
        x0 = y_init
        y_next = fsolve(self.myfun, x0)

        return self.rho_next, y_next

    def myfun(self, y_next):
        """Help fsolve update our relative flow data.

        Parameters
        ----------
        y_next : array_like
            array whose values to be determined

        Returns
        -------
        array_like or tuple
            functions to be minimized/maximized based on initial values
        """
        func = y_next + ((self.dt / self.tau) *
                         (self.rho_next * self.u(self.rho_next, y_next)) -
                         self.rhs)

        return func

    def ve(self, density):
        """Implement the 'Greenshields model for the equilibrium velocity'.

        Fan, Shimao et al. “Comparative model accuracy of a data-fitted
        generalized Aw-Rascle-Zhang model.” NHM 9 (2014): 239-268.

        Parameters
        ----------
        density : array_like
            density data at every specified point on road

        Returns
        -------
        array_like
            equilibrium velocity at every specified point on road
        """
        return self.v_max * (1 - (density / self.rho_max))

    def u(self, density, y_value):
        """Calculate actual velocity from density and relative flow.

        Parameters
        ----------
        density : array_like
            density data at every specified point on road
        y_value : array_like
            relative flow data at every specified point on road

        Returns
        -------
        array_like
            velocity at every specified point on road
        """
        return (y_value / density) + self.ve(density)

    def relative_flow(self, density, velocity):
        """Calculate actual relative flow from density and  velocity.

        Parameters
        ----------
        density : array_like
            density data at every specified point on road
        velocity : array_like
            velocity at every specified point on road

        Returns
        -------
        array_like
            relative flow at every specified point on road
        """
        return (velocity - self.ve(density)) * density
