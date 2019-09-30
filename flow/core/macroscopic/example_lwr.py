import numpy as np
import matplotlib.pyplot as plt
from flow.core.macroscopic import LWR
from flow.core.macroscopic.lwr import PARAMS as LWR_PARAMS

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

params = LWR_PARAMS.copy()


# Length of road
length = 35
# Spacial Grid Resolution
grid_resolution = 150
CFL = 0.95
dx = length / grid_resolution
x = np.arange(0, length, dx)
# initial density Points
U = initial(x)

params["initial_conditions"] = U
params["boundary_conditions"] = (0, 0)

if __name__ == "__main__":

    env = LWR(params)
    # run a single roll out of the environment
    obs = env.reset()

    # set the directory to save csv file.
    # results_dir = "~/flow/flow/models/macro/results/"
    # data = {"Position_on_street": x}

    for i in range(50):

        action = 1 # agent.compute(obs)
        obs, rew, done, _ = env.step(action)

        # store in dictionary and write to csv
        # new_densities = {"Densities at t = " + str(i): env.obs}
        # new_speeds = {"Velocities at t = " + str(i): env.speed_info()}
        # data.update(new_densities)
        # data.update(new_speeds)
        density = env.obs[:int(env.length/env.dx)]
        speeds = env.obs[int(env.length/env.dx):]
        plot_points(length, x, density, speeds, 4, 1)

    # write to csv file
    # path_for_results = results_dir + "lwr_results.csv"
    # pd.DataFrame(data).to_csv(path_for_results, index=False)

