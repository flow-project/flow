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

    ("total_time", 66, float, "time horizon (in seconds)"),

    ("dt", 0.066, float, "time discretization (in seconds/step)"),

    # ======================================================================= #
    #                      Initial / boundary conditions                      #
    # ======================================================================= #

    ("initial_conditions", ([0], [0]), None,
     "tuple of (density, speed) initial conditions. Each element of the tuple"
     " must be a list of length int(length/dx)"),

    ("boundary_conditions", 'extend_both', None,
     "conditions at road left and right ends; should either dict or string"
     " ie. {'constant_both': ((density, speed),(density, speed) )}, constant value of both ends"
     "loop, loop edge values as a ring"
     "extend_both, extrapolate last value on both ends"
     ),
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
    rho_next : array_like
        the next density value
    rhs : float
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
            * boundary_conditions (string) or (dict): string  of either "loop" or "extend_both " or a
                dict of {"condition": ((density,speed), (density,speed))} specifying densities at left edge
                of road and right edge of road respectively as initial boundary conditions.
                Boundary conditions are important to maintain conservation laws
                because at each calculation of the flux, we lose information at the boundaries
                and so may we have to keep updating/specifying them for t=0 (unless in special cases).
        """
        super(ARZ, self).__init__(params)

        assert (params['length'] / params['dx']).is_integer(), \
            "The 'length' variable in params must be divisible by 'dx'."

        # assert (params['total_time'] / params['dt']).is_integer(), \
        #     "The 'total_time' variable in params must be divisible by 'dt'."

        assert params['rho_max'] <= params['rho_max_max'], \
            "The 'rho_max' must be less than or equal to 'rho_max_max'"

        assert params['v_max'] <= params['v_max_max'], \
            "The 'v_max' must be less than or equal to 'v_max_max'"

        assert 0 <= params['CFL'] <= 1, \
            "'CFL' variable must be between 0 and 1"

        assert params['dt'] <= params['CFL'] * params['dx']/params['v_max'], \
            "CFL condition not satisfied. Make sure dt <= CFL * dx / v_max"

        assert len(params['initial_conditions'][0]) == \
            int(params['length'] / params['dx']) or len(params['initial_conditions'][0][0]) == \
            int(params['length'] / params['dx']), \
            "Initial conditions must be a list of size: (length/dx)."

        assert len(params['initial_conditions'][1]) == \
            int(params['length'] / params['dx']) or len(params['initial_conditions'][1][0]) == \
            int(params['length'] / params['dx']), \
            "Initial conditions must be a list of size: length/dx."

        self.params = params.copy()
        self.length = params['length']
        self.dx = params['dx']
        self.rho_max = params['rho_max']
        self.rho_max_max = params['rho_max_max']
        self.v_max = params['v_max']
        self.v_max_max = params['v_max_max']

        if params['CFL'] is None:
            self.CFL = params['dt'] * self.v_max / self.dx
        else:
            self.CFL = params['CFL']
        if params['dt'] is None:
            self.dt = self.CFL * self.dx / self.v_max
        else:
            self.dt = params['dt']

        self.tau = params['tau']
        self.total_time = params['total_time']
        self.horizon = int(self.total_time / self.dt)
        self.initial_conditions_density = params['initial_conditions'][0]
        self.initial_conditions_velocity = params['initial_conditions'][1]

        # calculate relative flow (transform velocity (u) to relative flow (y))
        # lam is an exponent of the Green-shield velocity function
        self.lam = 1
        self.initial_conditions_relative_flow = self.relative_flow(
            self.initial_conditions_density, self.initial_conditions_velocity, self.v_max, self.rho_max)
        self.initial_conditions = (self.initial_conditions_density,
                                   self.initial_conditions_relative_flow)
        self.list_values = []
        self.boundaries = params["boundary_conditions"]
        self.boundary_left = None
        self.boundary_right = None

        # critical density defined by the Green-shield Model
        self.rho_critical = self.rho_max / 2
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

        # extract the densities and relative speed
        obs = self.obs.copy()

        rew, self.obs = self._step(obs, self.v_max, self.rho_max, self.CFL, self.tau)

        # compute the done mask
        done = self.num_steps >= self.horizon

        return self.obs, rew, done, {}

    def _step(self, obs, v_max, rho_max, cfl, tau):

        if len(np.shape(obs)) > 1:

            rho = obs[:int(len(obs)/2)]
            speed = obs[int(len(obs)/2):]
            relative_flow = self.relative_flow(rho, speed, v_max, rho_max)
            rho, rel_flow = self.arz_solve((rho, relative_flow), v_max, rho_max, cfl, tau)
        else:

            rho = obs[:int(obs.shape[0] / 2)]
            speed = obs[int(obs.shape[0] / 2):]
            relative_flow = self.relative_flow(rho, speed, v_max, rho_max)
            rho, rel_flow = self.arz_solve((rho, relative_flow), v_max, rho_max, cfl, tau)

        speed = self.speed = self.u(rho, rel_flow, v_max, rho_max)

        # compute the reward value
        rew = np.mean(np.square(speed - v_max) * rho)

        # compute the new observation
        return rew, np.concatenate((rho, speed))

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
        densities, relative_flows = self.initial_conditions
        # speeds = self.u(densities, relative_flows)
        speeds = self.initial_conditions_velocity
        self.obs = np.concatenate((densities,
                                   speeds))

        return self.obs

    def arz_solve(self, u_full, v_max, rho_max, cfl, tau):
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
        v_max : see parent class
        rho_max : see parent class
        cfl : see parent class
        tau : see parent class

        Returns
        -------
        array_like
            next density data points as calculated by the Semi-Implicit Godunov
            scheme
        array_like
            next relative flow data points as calculated by the Semi-Implicit
            Godunov scheme
        """
        # store loop boundary conditions for ring-like experiment
        if self.boundaries == "loop":

            if len(np.shape(u_full[0])) > 1:
                self.boundary_left = u_full[0][:, np.shape(u_full)[1] - 1], u_full[1][:, np.shape(u_full)[1] - 1]
                self.boundary_right = u_full[0][:, np.shape(u_full)[1] - 2], u_full[1][:, np.shape(u_full)[1] - 2]

            else:
                self.boundary_left = u_full[0][len(u_full[1]) - 1], u_full[1][len(u_full[1]) - 1]
                self.boundary_right = u_full[0][len(u_full[1]) - 2], u_full[1][len(u_full[1]) - 2]

        # full array with boundary conditions
        u_all = u_full

        # compute flux
        fp_higher_half, fp_lower_half, fy_higher_half, fy_lower_half, \
            rho_init, y_init = self.compute_flux(u_all, v_max, rho_max)

        # update new points
        new_points = self.arz_update_points(
            fp_higher_half,
            fp_lower_half,
            fy_higher_half,
            fy_lower_half,
            rho_init,
            y_init,
            cfl,
            v_max,
            rho_max,
            tau
        )

        # update loop boundary conditions for ring-like experiment
        if self.boundaries == "loop":
            if len(np.shape(u_full[0])) > 1:
                rho_array = np.insert(np.append(new_points[0][:, 1:np.shape(new_points[0])[1] - 1],
                                                self.boundary_right[0].reshape(len(self.boundary_right[0]), 1), axis=1),
                                      0, self.boundary_left[0], axis=1)
                y_array = np.insert(np.append(new_points[1][:, 1:np.shape(new_points[1])[1] - 1],
                                              self.boundary_right[1].reshape(len(self.boundary_right[1]), 1), axis=1),
                                    0, self.boundary_left[1], axis=1)

                new_points = (rho_array, y_array)

            else:
                new_points = (np.insert(np.append(new_points[0], self.boundary_right[0]),
                                        0, self.boundary_left[0]),
                              np.insert(np.append(new_points[1], self.boundary_right[1]),
                                        0, self.boundary_left[1]))

        # update boundary conditions by extending/extrapolating boundaries (reduplication)
        if self.boundaries == "extend_both":
            if len(np.shape(u_full[0])) > 1:
                self.boundary_left = u_full[0][:, 0], u_full[1][:, 0]
                self.boundary_right = u_full[0][:, np.shape(u_full)[1] - 1], u_full[1][:, np.shape(u_full)[1] - 1]
                rho_array = np.insert(np.append(new_points[0][:, 1:np.shape(new_points[0])[1] - 1],
                                                self.boundary_right[0].reshape(len(self.boundary_right[0]), 1), axis=1),
                                      0, self.boundary_left[0], axis=1)
                y_array = np.insert(np.append(new_points[1][:, 1:np.shape(new_points[1])[1] - 1],
                                              self.boundary_right[1].reshape(len(self.boundary_right[1]), 1), axis=1),
                                    0, self.boundary_left[1], axis=1)

                new_points = (rho_array, y_array)
            else:
                self.boundary_left = (new_points[0][0], new_points[1][0])
                self.boundary_right = (new_points[0][len(new_points[1]) - 1],
                                       new_points[1][len(new_points[1]) - 1])
                new_points = (np.insert(np.append(new_points[0], self.boundary_right[0]),
                                        0, self.boundary_left[0]),
                              np.insert(np.append(new_points[1], self.boundary_right[1]),
                                        0, self.boundary_left[1]))

        # update boundary conditions by keeping boundaries constant
        if type(self.boundaries) == dict:
            if list(self.boundaries.keys())[0] == "constant_both":

                if len(np.shape(u_full[0])) > 1:

                    self.boundary_left = self.boundaries["constant_both"][0]
                    self.boundary_right = self.boundaries["constant_both"][1]
                    ones = np.reshape(np.ones(len(u_full[0])), (len(u_full[0]), 1))

                    rho_array = np.insert(np.append(new_points[0][:, 1:np.shape(new_points[0])[1] - 1],
                                                    self.boundary_right[0] * ones,
                                                    axis=1),
                                          0, self.boundary_left[0] * ones, axis=1)
                    y_array = np.insert(np.append(new_points[1][:, 1:np.shape(new_points[1])[1] - 1],
                                                  self.boundary_right[1] * ones,
                                                  axis=1),
                                        0, self.boundary_left[1] * ones, axis=1)

                    new_points = (rho_array, y_array)
                else:

                    self.boundary_left = (self.boundaries["constant_both"][0][0],
                                          self.boundaries["constant_both"][0][1])
                    self.boundary_right = (self.boundaries["constant_both"][1][0],
                                           self.boundaries["constant_both"][1][1])
                    new_points = (np.insert(np.append(new_points[0], self.boundary_right[0]),
                                            0, self.boundary_left[0]),
                                  np.insert(np.append(new_points[1], self.boundary_right[1]),
                                            0, self.boundary_left[1]))
        return new_points

    def compute_flux(self, u_all, v_max, rho_max):
        """Implement the Flux Supply and Demand Model for flux funtion.

        'Lebacque, Jean-Patrick & Haj-Salem, Habib & Mammar, Salim. (2005).
        Second order traffic flow modeling: supply-demand analysis of the
        inhomogeneous riemann problem and of boundary conditions'

        Parameters
        ----------
        u_all : tuple -> (density, relative flow)
            density : array_like
                density data points on the road length with boundary conditions
            relative flow : array_ike
               relative flow data points on the road length with boundary
               conditions

            Note: at time = 0, U_all = initial density data
        v_max : see parent class
        rho_max : see parent class

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
        # hack
        mask = rho_full == 0
        rho_full[mask] = 0.0000000001

        y_full = u_all[1]

        rho_critical = rho_max / 2
        q_max = ((rho_critical * self.ve(rho_critical, v_max, rho_max)) + y_full)

        # demand
        d = (rho_full * self.ve(rho_full, v_max, rho_max) + y_full) * \
            (rho_full < rho_critical) + q_max * (rho_full >= rho_critical)

        # supply
        s = (rho_full * self.ve(rho_full, v_max, rho_max) + y_full) * \
            (rho_full > rho_critical) + q_max * (rho_full <= rho_critical)

        if len(np.shape(rho_full)) > 1:
            new_col = s[:, len(s) - 1].reshape(len(s), 1)
            s = np.concatenate((s[:, 1:], new_col), axis=1)
        else:
            s = np.append(s[1:], s[len(s) - 1])

        # flow flux
        q = np.minimum(d, s)

        # relative flux
        p = q * (y_full / rho_full)

        # fluxes at left cell boundaries
        if len(np.shape(rho_full)) > 1:
            qm = np.insert(q[:, 0:len(q[0]) - 1], 0, q[:, 0], axis=1)
            pm = np.insert(p[:, 0:len(p[0]) - 1], 0, p[:, 0], axis=1)

        else:
            qm = np.append(q[0], q[0:len(q) - 1])
            pm = np.append(p[0], p[0:len(p) - 1])

        fp_higher_half = q
        fp_lower_half = qm
        fy_higher_half = p
        fy_lower_half = pm
        rho_init = rho_full
        y_init = y_full

        return fp_higher_half, fp_lower_half, fy_higher_half, fy_lower_half, \
            rho_init, y_init

    def arz_update_points(self,
                          fp_higher_half,
                          fp_lower_half,
                          fy_higher_half,
                          fy_lower_half,
                          rho_init,
                          y_init,
                          cfl,
                          v_max,
                          rho_max,
                          tau):
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
        v_max : see parent class
        rho_max : see parent class
        cfl : see parent class
        tau : see parent class

        Returns
        -------
        array_like
            next density values at each point on the road
        array_like
            next relative flow values at each point on the road
        """
        # time and cell step
        # Hacked dt to explicitly depend CFL(see line 209) <-- this is helpful for calibration
        self.dt = cfl * self.dx / v_max

        step = self.dt / self.dx  # where are we referencing this?

        # updating density
        self.rho_next = rho_init + (step * (fp_lower_half - fp_higher_half))

        # updating relative flow
        # right hand side constant -> we use fsolve to find our roots
        self.rhs = y_init + (step * (fy_lower_half - fy_higher_half)) \
            + (self.dt / tau) * self.rho_next * self.ve(self.rho_next, v_max, rho_max)
        x0 = y_init
        y_next = fsolve(self.myfun, x0, args=(tau, v_max, rho_max))
        if len(np.shape(self.rho_next)) > 1:
            y_next = np.reshape(y_next, ((len(self.rho_next)), len(self.rho_next[0])))

        if len(np.shape(rho_init)) > 1:
            self.rho_next = self.rho_next[:, 1: len(self.rho_next[0]) - 1]
            y_next = y_next[:, 1: len(y_next[0]) - 1]

        else:
            self.rho_next = self.rho_next[1:len(self.rho_next) - 1]
            y_next = y_next[1:len(y_next) - 1]

        return self.rho_next, y_next

    def myfun(self, y_next, *args):
        """Help fsolve update our relative flow data.

        Parameters
        ----------
        y_next : array_like
            array whose values to be determined
        args :  v_max : see parent class
                rho_max : see parent class
                cfl : see parent class
                tau : see parent class

        Returns
        -------
        array_like or tuple
            functions to be minimized/maximized based on initial values
        """
        tau, v_max, rho_max = args
        if len(np.shape(self.rho_next)) > 1:
            y_next = np.reshape(y_next, ((len(self.rho_next)), len(self.rho_next[0])))

        func = y_next + ((self.dt / tau) *
                         (self.rho_next * self.u(self.rho_next, y_next, v_max, rho_max)) -
                         self.rhs)

        if len(np.shape(self.rho_next)) > 1:
            return np.reshape(func, (len(self.rho_next[0]) * len(self.rho_next),))

        else:
            return func

    def ve(self, density, v_max, rho_max):
        """Implement the 'Greenshields model for the equilibrium velocity'.

        Fan, Shimao et al. “Comparative model accuracy of a data-fitted
        generalized Aw-Rascle-Zhang model.” NHM 9 (2014): 239-268.

        Parameters
        ----------
        density : array_like
            density data at every specified point on road
        v_max : see parent class
        rho_max : see parent class

        Returns
        -------
        array_like
            equilibrium velocity at every specified point on road
        """
        return v_max * ((1 - (density / rho_max)) ** self.lam)

    def u(self, density, y_value, v_max, rho_max):
        """Calculate actual velocity from density and relative flow.
            Note: if density is zero, make density = 0.0000001

        Parameters
        ----------
        density : array_like
            density data at every specified point on road
        y_value : array_like
            relative flow data at every specified point on road
        v_max : see parent class
        rho_max : see parent class
        
        Returns
        -------
        array_like
            velocity at every specified point on road
        """
        # hack
        mask = density == 0
        density[mask] = 0.0000000001

        return (y_value / density) + self.ve(density, v_max, rho_max)

    def relative_flow(self, density, velocity, v_max, rho_max):
        """Calculate actual relative flow from density and  velocity.

        Parameters
        ----------
        density : array_like
            density data at every specified point on road
        velocity : array_like
            velocity at every specified point on road
        v_max : see parent class
        rho_max : see parent class

        Returns
        -------
        array_like
            relative flow at every specified point on road
        """
        return (velocity - self.ve(density, v_max, rho_max)) * density
