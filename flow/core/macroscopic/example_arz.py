import numpy as np
import matplotlib.pyplot as plt
from flow.core.macroscopic import ARZ
from flow.core.macroscopic.arz import PARAMS as ARZ_PARAMS, boundary_left_solve, boundary_right_solve

params = ARZ_PARAMS.copy()

length = 10
grid_resolution = 150
dx = length / grid_resolution
x = np.arange(0.5 * dx, (length - 0.5 * dx), dx)

# density initial_data
rho_L_side = 0.5 * (x < max(x) / 2)
rho_R_side = 0.5 * (x > max(x) / 2)
u_data_rho_rho = rho_L_side + rho_R_side
# velocity initial_data
u_L_side = 0.7 * (x < max(x) / 2)
u_R_side = 0.1 * (x > max(x) / 2)
u_data_rho_velocity = u_L_side + u_R_side

params["dx"] = dx
params['initial_conditions'] = (u_data_rho_rho, u_data_rho_velocity)
params['boundary_conditions'] = boundary_left_solve(params['initial_conditions']), \
                                boundary_right_solve(params['initial_conditions'])

env = ARZ(params)

# run a single roll out of the environment
obs = env.reset()

# uncomment to store the data
# results_dir = "~/flow/flow/models/macro/results/"
# data = {"Position_on_street": x}

if __name__ == '__main__':
    for i in range(50):
        action = 1
        obs, rew, done, _ = env.step(action)

        # uncomment to store the data
        # new_densities = {"Densities at t = " + str(i): env.obs[0]}
        # new_relative_flow = {"Relative flow at t = " + str(i): env.obs[1]}
        # new_speeds = {"Velocities at t = " + str(i): u(env.obs[0], env.obs[1], V_max, rho_max)}
        # data.update(new_densities)
        # data.update(new_speeds)
        # data.update(new_relative_flow)

        # plot current profile during execution
        plt.plot(x, env.obs[0], 'b-')
        plt.axis([0, length, 0.4, 0.8])
        plt.ylabel('Density')
        plt.xlabel('Street Length')
        plt.title("ARZ Evolution of Density")
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    # final plot
    plt.plot(x, env.obs[0], 'b-')
    plt.axis([0, length, 0.4, 0.8])
    plt.ylabel('Density')
    plt.xlabel('Street Length')
    plt.title("ARZ Evolution of Density")
    plt.show()

    # uncomment save file to csv
    # path_for_results = results_dir + "arz_results.csv"
    # pd.DataFrame(data).to_csv(path_for_results, index=False)
