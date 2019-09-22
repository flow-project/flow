"""Fill in . todo."""
import numpy as np
import matplotlib.pyplot as plt
from flow.core.macroscopic.lwr import LWR


def plot_points(length, x, density, speed, rho_max, v_max):
    """Fill in . todo."""
    # plot current profile during execution
    fig, plots = plt.subplots(2)
    plots[0].plot(x, density, 'b-')
    plots[0].axis([0, length, -0.1, rho_max + 1])
    plots[0].set(xlabel='Street Length (m)', ylabel='Density')
    plots[0].set_title("LWR Evolution of Density")

    plots[1].plot(x, speed, 'b-')
    plots[1].axis([0, length, -0.1, v_max + 1])
    plots[1].set(xlabel='Street Length (m)', ylabel='Velocities(m/s)')
    plots[1].set_title("LWR Evolution of Velocities ")
    plt.show()


def initial(points):
    """Calculate the initial density at each specified point on the road.

    Note: This is a calibration that's used just an example.

    Parameters
    ----------
    points : array_like
        points on the road length from 0 to road length

    Returns
    -------
    values : array_like
            calculated initial density data
    """
    values = 1 * (points <= 5) + (-4 + points) * (points > 5) \
        * (points <= 6) + 2 * (points > 6) * (points < 15) \
        + (2 * points - 28) * (points > 15) * (points <= 16) \
        + 4 * (points > 16) * (points < 25) + 1 * (points >= 25)

    return values


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

    # specify time horizon
    iterations = 50

    env = LWR(
        initial_conditions=U,
        length=L,
        boundary_data=(u_l, u_r),
        max_density=R,
        V_max=V_max
    )

    # run a single roll out of the environment
    obs = env.reset()

    for _ in range(int(iterations)):
        action = V_max  # agent.compute(obs)
        obs, rew, done, _ = env.step(action)
        plot_points(L, x, env.obs, env.speed_info(), R, V_max)
