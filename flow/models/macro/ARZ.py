import gym
from scipy.optimize import fsolve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def boundary_right(data):
    """Calculate right boundary condition

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

    # RIGHT boundary condition
    return data[0][len(data[1]) - 1], data[1][len(data[1]) - 1]


def boundary_left(data):
    """Calculate left boundary condition

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

    # Left boundary condition
    return data[0][0], data[1][0]


def ARZ_Solve(u_full, u_right, u_left):
    """Implement Godunov Semi-Implicit scheme for multi-populations.

    Fan, Shimao et al. “Comparative model accuracy of a data-fitted generalized Aw-Rascle-Zhang model.”
    NHM 9 (2014): 239-268.

    Parameters
    ----------
    u_full: tuple -> (density, relative flow)
        density: array_like
            density data points on the road length

        relative flow: array_ike
           relative flow data points on the road length

        Note: at time = 0, u_full = initial density data

    u_right:  tuple --> (double, double)
        right boundary conditions for Data -> (density, relative flow)
    u_left:  tuple --> (double, double)
        left boundary conditions for Data -> (density, relative flow)
    Returns
    -------
    new_points: tuple -> (density, relative flow)
        density: array_like
            next density data points as calculated by the Semi-Implicit Godunov scheme

        relative flow: array_ike
           next relative flow data points as calculated by the Semi-Implicit Godunov scheme

    """
    # We shouldn't have this in here (Need to find a better way to update left and right data(or keep 'em constant)
    u_left = boundary_left(u_full)
    u_right = boundary_right(u_full)

    # time and cell step
    step = dt / dx

    # full arrray with boundary conditions
    u_all = np.insert(np.append(u_full[0], u_right[0]), 0, u_left[0]), np.insert(
        np.append(u_full[1], u_right[1]), 0, u_left[1])

    # compute flux
    Fp_higher_half, Fp_lower_half, Fy_higher_half, Fy_lower_half, rho_init, y_init = Compute_Flux(u_all,dt, dx)

    # update new points
    new_points = ARZ_update_points(Fp_higher_half, Fp_lower_half, Fy_higher_half, Fy_lower_half, rho_init, y_init, Ve, tau, dt, step)

    return new_points


def Function_rho(density, y_value):
    """ Calculate flux associated with density using density Flux function

    Fan, Shimao et al. “Comparative model accuracy of a data-fitted generalized Aw-Rascle-Zhang model.”
    NHM 9 (2014): 239-268.

    Parameters
    ------------
    density: array_like
        density data points at every specified point on the road

    y_value: array_like
        relative flow data points at every specified point on the road

    Returns
    -----------
    array_like:
        density flux values

    """
    #Flux equation for density (rho)
    return y_value + (density * Ve(density))


def Function_y(density, y_value):
    """ Calculate flux associated with relative flow using Relative Flow Flux function

    Fan, Shimao et al. “Comparative model accuracy of a data-fitted generalized Aw-Rascle-Zhang model.”
    NHM 9 (2014): 239-268.

    Parameters
    ------------
    density: array_like
        density data points at every specified point on the road

    y_value: array_like
        relative flow data points at every specified point on the road

    Returns
    -----------
    array_like:
        relative flow flux values
    """

    return ((y_value ** 2) / density) + (y_value * Ve(density))


def Compute_Flux(U_all,dt, dx):
    """ Implement the 'The Lax-Friedrichs Method' for Flux calculations for each cell

    "Finite-Volume Methods for Hyperbolic Problems", Chapter 4 (p71), Randal J. Leveque.

    Parameters
    ------------
    U_all: tuple -> (density, relative flow)
        density: array_like
            density data points on the road length with boundary conditions

        relative flow: array_ike
           relative flow data points on the road length with boundary conditions

        Note: at time = 0, U_all = initial density data

    dt: double
        change in time

    dx: double
        change in length

    Returns
    -----------
    tuple -> (fp_higher_half, fp_lower_half, fy_higher_half, fy_lower_half, rho_init, y_init)
        fp_higher_half: array_like
            density flux at right boundary of each cell

        fp_lower_half: array_like
            density flux at left boundary of each cell

        fy_higher_half: array_like
            relative flow flux at right boundary of each cell

        fy_lower_half: array_like
            relative flow flux at left boundary of each cell

        rho_init: array_like
            current values for density at each point on the road (midpoint of cell)

        y_init: array_like
            current values for relative flow at each point on the road (midpoint of cell)

"""
    rho_full = U_all[0]
    y_full = U_all[1]

    # left cell boundary data to be considered -> entire row except last two
    rho_l = rho_full[:-2]
    y_l = y_full[:-2]

    # midpoint cell boundary data to be considered -> all expect first and last
    rho_init = rho_full[1:-1]
    y_init = y_full[1:-1]

    # right cell boundary data to be considered -> entire row except first two
    rho_r = rho_full[2:]
    y_r = y_full[2:]

    # left fluxes
    fp_lower_half = 0.5 * (Function_rho(rho_l, y_l) + Function_rho(rho_init, y_init)) - (
            (0.5 * dt / dx) * (rho_init - rho_l))
    fy_lower_half = 0.5 * (Function_y(rho_l, y_l) + Function_y(rho_init, y_init)) - ((0.5 * dt / dx) * (y_init - y_l))

    # right fluxes
    fp_higher_half = 0.5 * (Function_rho(rho_r, y_r) + Function_rho(rho_init, y_init)) - (
            (0.5 * dt / dx) * (rho_r - rho_init))
    fy_higher_half = 0.5 * (Function_y(rho_r, y_r) + Function_y(rho_init, y_init)) - ((0.5 * dt / dx) * (y_r - y_init))

    return fp_higher_half, fp_lower_half, fy_higher_half, fy_lower_half, rho_init, y_init



def ARZ_update_points (fp_higher_half, fp_lower_half, fy_higher_half, fy_lower_half, rho_init, y_init, Ve, tau, dt,step):
    """ Update our current density and relative flow values

    Parameters
    ------------
    fp_higher_half: array_like
        density flux at right boundary of each cell

    fp_lower_half: array_like
        density flux at left boundary of each cell

    fy_higher_half: array_like
        relative flow flux at right boundary of each cell

    fy_lower_half: array_like
        relative flow flux at left boundary of each cell

    rho_init: array_like
        current values for density at each point on the road (midpoint of cell)

    y_init: array_like
        current values for relative flow at each point on the road (midpoint of cell)

    Returns
    -----------
    tuple -> (rho_next, y_next)
        rho_next: array_like
            next density values at each point on the road
        y_next: array_like
            next relative flow values at each point on the road
    """
    # updating density
    global rho_next
    rho_next = rho_init + (step * (fp_lower_half - fp_higher_half))

    # updating relative flow
    # right hand side constant -> we use fsolve to find our roots
    global rhs
    rhs = y_init + (step * (fy_lower_half - fy_higher_half)) + ((dt / tau) * ((rho_next * Ve(rho_next))))
    x0 = y_init
    y_next = fsolve(myfun, x0)

    return rho_next, y_next


def myfun(y_next):
    """helper function to help fsolve update our relative flow data

    Parameters
    ------------
    y_next: array_like
        array whose values to be determined

    Returns
    -----------
    func: array_like or tuple
        functions to be minimized/maximized based on initial values
    """

    func = y_next + ((dt / tau) * (rho_next * u(rho_next, y_next)) - rhs)
    return func


def Ve(density):
    """ Implement the 'Greenshields model for the equilibrium velocity'

    Fan, Shimao et al. “Comparative model accuracy of a data-fitted generalized Aw-Rascle-Zhang model.”
    NHM 9 (2014): 239-268.

    Parameters
    ------------
    density: array_like
        density data at every specified point on road

    Returns
    ------------
    array_like
        equilibrium velocity at every specified point on road
    """

    return u_max * (1 - (density / rho_max))


def u(density, y_value):
    """ Calculate actual velocity from density and relative flow

    Parameters
    ------------
    density: array_like
        density data at every specified point on road
    y_value: array_like
        relative flow data at every specified point on road

    Returns
    ------------
    array_like
        velocity at every specified point on road
    """

    return (y_value / density) + Ve(density)  # velocity function


class ARZ(gym.Env):
    """ Create an rl environment to train and test the ARZ model

    Attributes
    ----------
    init : tuple -> (density, relative flow)
        density: array_like
            density data points on the road length with boundary conditions

        relative flow: array_ike
           relative flow data points on the road length with boundary conditions

        Note: at time = 0, init = initial density data
    boundary : tuple -> (double, double)
        left boundary condition
    obs : array_like
        the most recent observation
    """

    def __init__(self, initial_conditions, boundary_left):
        """Initialize the LWR model.
           Parameters
           ----------
           initial_conditions : tuple -> (density, relative flow)
               initial density data and relative flow on the road length at time = 0
           boundary_left : double
               left boundary condition
       """

        self.init = initial_conditions
        self.boundary = boundary_left
        self.obs = initial_conditions

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

        obs = []
        rew = 0
        done = False
        info_dict = {}

        # advance the state of the simulation by one step
        self.obs = ARZ_Solve(self.obs, rl_actions, self.boundary)

        return obs, rew, done, info_dict

    def reset(self):
        """Reset the environment.
        Returns
        -------
        tuple -> (array_like, array_like)
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """

        self.obs = self.init

        return self.obs


if __name__ == '__main__':

    global L, N, dx, CFL, dt, x, tau, rho_max, u_max, u_r, u_l
    # define variables, parameters and functions
    # PARAMETERS
    rho_max = 1  # maximum_density
    u_max = 1  # maximum velocity
    #  length of road
    L = 1
    N = 100  # spacial grid resolution /  cell space should be atleast n = 300
    dx = L / N
    # CFL condition--must becloser to or equal to 1 (dictactes the speed of information travel)
    CFL = 0.99
    dt = CFL * dx / u_max
    # scaling -- points on street we are plotting against
    x = np.arange(0.5 * dx, (L - 0.5 * dx), dx)
    # time needed to adjust velocity from u to Ve
    tau = 0.1

    ########
    # initial_data
    # Density
    rho_L_side = 0.5 * (x < max(x) / 2)
    rho_R_side = 0.5 * (x > max(x) / 2)

    # Velocity
    u_L_side = 0.7 * (x < max(x) / 2)
    u_R_side = 0.1 * (x > max(x) / 2)

    u_data_rho_rho = rho_L_side + rho_R_side  # density
    u_data_rho_velocity = u_L_side + u_R_side  # velocity
    # Calculate relative flow
    y_vector = (u_data_rho_rho * (u_data_rho_velocity - Ve(u_data_rho_rho)))



    initial_data = u_data_rho_rho, y_vector
    # Boundary conditions
    u_l = boundary_left(initial_data)
    u_r = boundary_right(initial_data)

    env = ARZ(initial_data, u_l)

    obs = env.reset()

    for _ in range(50):
        action = u_r
        obs, rew, done, _ = env.step(action)
        # density plot
        # our initial data vector = [density ; relative flow]
        plt.plot(x, env.obs[0], 'b-')
        plt.axis([0, L, 0.4, 0.8])
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    # final plot
    plt.plot(x, env.obs[0], 'b-')
    plt.axis([0, L, 0.4, 0.8])
    plt.show()
