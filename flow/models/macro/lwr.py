import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def initial(x):
    """TODO

    Parameters
    ----------
    x : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    values = 1 * (x <= 5) + (-4 + x) * (x > 5) * (x <= 6) + 2 * (x > 6) * (
                x < 15) + (2 * x - 28) * (x > 15) * (x <= 16) + 4 * (
                         x > 16) * (x < 25) + 1 * (x >= 25)
    return values


def Gflux(u, v, r):
    """TODO

    Parameters
    ----------

    Returns
    -------
    TODO
        TODO
    """
    # demand
    d = v * u * (1 - u / r) * (u < 0.5 * r) + 0.25 * v * r * (U >= 0.5 * r)

    # supply
    s = v * u * (1 - u / r) * (u > 0.5 * r) + 0.25 * v * r * (u <= 0.5 * r)
    s = np.append(s[1:], s[len(s) - 1])

    # Godunov flux
    return np.minimum(d, s)


def IBVP(u, u_r, u_l):
    """Implement Godunov scheme for multi-populations.

    P.Goatin, June 2017 TODO: replace with full citation

    Parameters
    ----------
    u : TODO
        TODO
    u_r : TODO
        TODO
    u_l : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    #
    # P.Goatin, June 2017
    # clf
    # """"parameters"""
    L = 30  # length of road
    N = 0.5  # (N = L/dx)reduce spacial grid resolution
    dx = L / N

    x = np.array([1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5])

    V = 1
    R = 4

    CFL = 0.95
    dt = CFL * dx / V
    lam = dt / dx

    u = np.insert(np.append(u, u_r), 0, u_l)

    # Godunov numerical flux
    F = Gflux(u, V, R)

    Fm = np.insert(F[0:len(F) - 1], 0, F[0])

    # Godunov scheme  (UPDATING U)
    u = u - lam * (F - Fm)

    u = np.insert(np.append(u[1:len(u) - 1], u_r), 0, u_l)

    # plot current profile during execution
    plt.plot(x, u[1:len(u) - 1], 'b-')
    plt.axis([0, L, -0.1, 4.1])
    plt.show()

    return u[1:len(u) - 1]


class LWR(gym.Env):
    """TODO

    TODO

    Attributes
    ----------
    init : TODO
        TODO
    boundary : TODO
        TODO
    obs : array_like
        the most recent observation
    """

    def __init__(self, initial_conditions, boundary_left):
        """Initialize the LWR model.

        Parameters
        ----------
        initial_conditions : TODO
            TODO
        boundary_left : TODO
            TODO
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
        self.obs = IBVP(self.obs, rl_actions, self.boundary)

        return obs, rew, done, info_dict

    def reset(self):
        """Reset the environment.

        Returns
        -------
        array_like
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        self.obs = self.init

        return self.obs


if __name__ == "__main__":
    # A few more parameters we need to preset
    #
    # Length of road and spacial grid resolution --> this will be important
    # when calibrating

    # points on x axis to plot
    x = np.array([1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5])

    # compute initial points
    U = initial(x)

    # initialize the right and left boundary conditions
    u_r = 0
    u_l = 0

    env = LWR(U, u_l)

    # run a single rollout of the environment
    obs = env.reset()
    for _ in range(10):
        action = u_r  # agent.compute(obs)
        obs, rew, done, _ = env.step(action)
        # update plot
