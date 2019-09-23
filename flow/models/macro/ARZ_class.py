"""Fill in . TODO."""
import gym
import numpy as np
import matplotlib.pyplot as plt


class ARZ(gym.Env):
    """Create an rl environment to train and test the ARZ model.

    Attributes
    ----------
    init : tuple -> (density, relative flow)
        density: array_like
            density data points on the road length with boundary conditions
        relative flow: array_ike
           relative flow data points on the road length with boundary
           conditions
        Note: at time = 0, init = initial density data
    boundary : tuple -> (double, double)
        left boundary condition
    obs : array_like
        the most recent observation
    """

    def __init__(self,
                 initial_data,
                 length,
                 boundary_data,
                 max_density,
                 tau,
                 V_max):
        """Initialize the LWR model.

        Parameters
        ----------
        initial_conditions : tuple -> (density, relative flow)
            initial density data and relative flow on the road length at time=0
        boundary_left : double
            left boundary condition
        """
        self.init = initial_data
        self.boundary_left = boundary_data[0]
        self.boundary_right = boundary_data[1]
        self.obs = initial_data
        self.length = length
        self.v_max = V_max
        self.rho_max = max_density
        self.tau = tau

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
        done = False

        if rl_actions is not None:
            self.v_max = rl_actions

        # advance the state of the simulation by one step
        self.obs = self.arz_solve(self.obs, self.tau, self.v_max, self.rho_max)

        speed = self.u(self.obs[0], self.obs[1], self.v_max, self.rho_max)

        rew = np.mean(speed) < self.v_max

        return self.obs.copy(), rew, done, {}

    def reset(self):
        """See parent class."""
        self.obs = self.init
        self.v_max = 27.5

        return self.obs


if __name__ == '__main__':
    """Run our environment.

    Parameters to be set are:
    -------------------------
    L : double
        Length of Road
    N : double
        Spacial Grid Resolution (number of points we need on street)
    CFL: double
        CFL Condition (dictates the speed of information travel)
        Note: must be between 0 and 1
    u_max: double
        maximum velocity
    rho_max: double
        maximum density
    U: (array_like, array_like)
        initial data
    u_r: (double, double)
        right boundary data
    u_l: (double, double)
        left boundary condition
    tau: double
        time needed to adjust velocity from u to Ve (equilibrium Velocity)
    """
    # maximum velocity
    V_max = 27.5

    # length of road
    L = 100000

    # spacial grid resolution /  cell space should be atleast n = 300
    N = 1500

    # CFL condition
    CFL = 0.99

    # change in length and points on road we are plotting against
    dx = L / N

    # maximum_density
    rho_max = dx/5

    x = np.arange(0.5 * dx, (L - 0.5 * dx), dx)

    # dt = change in time
    dt = CFL * dx / V_max

    # time needed to adjust velocity from u to Ve
    tau = 0.1

    # density initial_data
    rho_L_side = 0.5 * (x < max(x) / 2)
    rho_R_side = 0.5 * (x > max(x) / 2)

    # velocity initial_data
    u_L_side = 0.7 * (x < max(x) / 2)
    u_R_side = 0.1 * (x > max(x) / 2)

    u_data_rho_rho = rho_L_side + rho_R_side
    u_data_rho_velocity = u_L_side + u_R_side

    # calculate relative flow (transform u to y)
    y_vector = u_data_rho_rho * (u_data_rho_velocity -
                                 ve(u_data_rho_rho, V_max, rho_max))

    # full initial data
    initial_data = u_data_rho_rho, y_vector

    # right and left boundary conditions
    u_l = boundary_left(initial_data)
    u_r = boundary_right(initial_data)

    env = ARZ(
        initial_data=initial_data,
        length=L,
        boundary_data=(u_l, u_r),
        tau=tau,
        V_max=V_max,
        max_density=rho_max
    )

    # run a single roll out of the environment
    obs = env.reset()

    for _ in range(50):
        action = 1  # agent.compute(obs)
        obs, rew, done, _ = env.step(action)

        # plot current profile during execution
        plt.plot(x, env.obs[0], 'b-')
        plt.axis([0, L, 0.4, 0.8])
        plt.ylabel('Density')
        plt.xlabel('Street Length')
        plt.title("ARZ Evolution of Density")
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    # final plot
    plt.plot(x, env.obs[0], 'b-')
    plt.axis([0, L, 0.4, 0.8])
    plt.ylabel('Density')
    plt.xlabel('Street Length')
    plt.title("ARZ Evolution of Density")
    plt.show()
