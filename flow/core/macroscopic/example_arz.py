"""Example ARZ runner script."""
import numpy as np
from flow.core.macroscopic import ARZ
from flow.core.macroscopic.arz import PARAMS as ARZ_PARAMS
from flow.core.macroscopic.utils import run


def main():
    """Run the ARZ model and visualize the results."""
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
    params["total_time"] = 66.0

    params['boundary_conditions'] = "loop"
    # params['boundary_conditions'] = "extend_both"
    # params['boundary_conditions'] = {"constant_both": ((0.45, 0), (0.56, 1))}

    env = ARZ(params)
    run(env, rl_actions=1, visualize=True)


if __name__ == "__main__":
    main()
