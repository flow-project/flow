import gym
import numpy as np
import matplotlib.pyplot as plt

def plot_points(Length, x , Density, Speed, R, V_max):
    # plot current profile during execution
    fig, plots = plt.subplots(2)
    plots[0].plot(x, Density, 'b-')
    plots[0].axis([0, Length, -0.1, R + 1])
    plots[0].set(xlabel='Street Length (m)', ylabel='Density')
    plots[0].set_title("LWR Evolution of Density")

    plots[1].plot(x, Speed, 'b-')
    plots[1].axis([0, Length, -0.1, V_max + 1])
    plots[1].set(xlabel='Street Length (m)', ylabel='Velocities(m/s)')
    plots[1].set_title("LWR Evolution of Velocities ")
    plt.show()
    # plt.pause(0.00001)
    # plt.clf()


def initial(points):
    """Note: This is a calibration that's used just an example"""

    """ Calculate the initial density data at each
        specified point on the road length

    Parameters
    ----------
    points : array_like
        points on the road length from 0 to road length

    Returns
    -------
    values: array_like
            calculated initial density data
    """

    values = 1 * (points <= 5) + (-4 + points) * (points > 5) * (points <= 6) + 2 * (points > 6) * (
            points < 15) + (2 * points - 28) * (points > 15) * (points <= 16) + 4 * (
                     points > 16) * (points < 25) + 1 * (points >= 25)
    return values


def Gflux(u, v, r):
    """ Calculate the flux vector of our data

    Parameters
    ----------
    u: array_like
       data containing boundary conditions to be analysed
    v: double
       maximum velocity
    r: double
       maximum density

    Returns
    -------
    array_like
        array of fluxes calibrated at every point of our data
    """
    # demand
    d = v * u * (1 - u / r) * (u < 0.5 * r) + 0.25 * v * r * (u >= 0.5 * r)

    # supply
    s = v * u * (1 - u / r) * (u > 0.5 * r) + 0.25 * v * r * (u <= 0.5 * r)
    s = np.append(s[1:], s[len(s) - 1])

    # Godunov flux
    return np.minimum(d, s)


def IBVP(u, u_right, u_left, V_max, dt):
    """Implement Godunov scheme for multi-populations.

    Friedrich, Jan & Kolb, Oliver & Goettlich, Simone. (2018).
    A Godunov type scheme for a class of LWR traffic flow models with non-local flux.
    Networks & Heterogeneous Media. 13. 10.3934/nhm.2018024.

    Parameters
    ----------
    u: array_like
        density data to be analyzed and calculate next points for this data using Godunov scheme.
        Note: at time = 0, u = initial density data
    u_right: double
        right boundary condition
    u_left: double
        left boundary condition

    Returns
    -------
    array_like
          next density data point points as culculated by the Godunov scheme
    """

    # lam = time/distance step
    lam = dt / dx

    u = np.insert(np.append(u, u_right), 0, u_left)

    # Godunov numerical flux
    f = Gflux(u, V_max, R)

    fm = np.insert(f[0:len(f) - 1], 0, f[0])

    # Godunov scheme  (updating u)
    u = u - lam * (f - fm)

    u = np.insert(np.append(u[1:len(u) - 1], u_right), 0, u_left)

    return u[1:len(u) - 1]


class LWR(gym.Env):
    """ Create an rl environment to train and test the lwr model

    Attributes
    ----------
    init : array_like
        density data to be analyzed.
        Note: at time = 0, init = initial density data
    boundary_left : double
        left boundary condition
    obs : array_like
        the most recent observation
    """

    def __init__(self, length, initial_conditions, boundary_data, V_max, max_density):
        """Initialize the LWR model.

        Parameters [TO BE UPDATED]
        ----------
        initial_conditions : array_like
            initial density data on the road length at time = 0
        boundary_left : double
            left boundary condition
        """
        self.init = initial_conditions
        self.boundary_left = boundary_data[0]
        self.boundary_right = boundary_data[1]
        self.obs = initial_conditions
        self.length = length
        self.v_max = V_max
        self.rho_max = max_density


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
        done = False
        info_dict = {}


        # if rl_action = None:
        #     Initialize rl_actions (self.v_max) to some value
        if rl_actions == None:
            # advance the state of the simulation by one step
            self.obs = IBVP(self.obs, self.boundary_right, self.boundary_left, V_max=self.v_max, dt=dt)

        else:
            self.v_max = rl_actions
            self.obs = IBVP(self.obs, self.boundary_right, self.boundary_left, V_max=self.v_max, dt=dt)

        # check if reward is 0
        # if so, do nothing and advance to the next state with the same previous V_max
        # if reward is 1
        # set the new vmax to max(speeds)
        # advance to next state

        #if the mean speed is less than the previous v_max,
        # set v_max to the max of the current speeds
        rew = np.mean(self.speed_info()) < self.v_max

        # if rew == 1:
        #     self.v_max = np.max(self.speed_info())

        return obs, rew, done, info_dict

    def speed_info(self):
        """ Implement the 'Greenshields model for the equilibrium velocity'
        Fan, Shimao et al. “Comparative model accuracy of a data-fitted generalized Aw-Rascle-Zhang model.”
        NHM 9 (2014): 239-268.
        # ------------
        # array_like
        #     equilibrium velocity at every specified point on road
        """
        return self.v_max * (1 - (self.obs / self.rho_max))


    def reset(self):
        """Reset the environment.

        Returns
        -------
        array_like
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        self.v_max = 27.5
        self.obs = self.init

        return self.obs


if __name__ == "__main__":

    """ Run our environment.

    Parameters to be set are:
    -------------------------
    L : double
        Length of Road    
    N : double
        Spacial Grid Resolution (number of points we need on street)
    CFL: double
        CFL Condition (dictates the speed of information travel)
        Note: must be between 0 and 1
    V: double
        maximum velocity
    R: double
        maximum density
    U: array_like
        initial data
    u_r: double
        right boundary condition
    u_l: double
        left boundary condition

    ?? Time: double
        specify time horizon in seconds (is theoretical and depends on speed of code (dt, N and CFL) )
    """
    # Length of road
    L = 100000

    # Spacial Grid Resolution
    N = 1500

    # CFL condition
    CFL = 0.95

    # maximum velocity and maximum Density

    V_max = 27.5
    # R = dx/5
    # change in length and points on road we are plotting against
    dx = L / N
    R = dx/5
    x = np.arange(0.5 * dx, (L - 0.5 * dx), dx)

    # dt = change in time
    dt = CFL * dx / V_max
    # initial density Points
    U = initial(x)

    # right and left boundary conditions
    u_r = 0
    u_l = 0

    # #specify time horizon
    # Time = 15
    # specify time horizon #Time/dt find a more accurate way of finding the time should be dt/

    iterations = 50
    env = LWR(initial_conditions=U, length=L, boundary_data=(u_l, u_r), max_density=R, V_max= V_max)

    # run a single roll out of the environment
    obs = env.reset()

    for _ in range(int(iterations)):

        action = V_max # agent.compute(obs)
        obs, rew, done, _ = env.step(action)
        plot_points(L, x, env.obs, env.speed_info(), R, V_max)

