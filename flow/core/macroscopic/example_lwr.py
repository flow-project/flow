"""Example LWR runner script."""
import numpy as np
from flow.core.macroscopic import LWR
from flow.core.macroscopic.lwr import PARAMS as LWR_PARAMS
from flow.core.macroscopic.utils import run


def initial(points):
    """Calculate the initial density at each specified point on the road.

    Note: This is a calibration that's used just an example.

    Parameters
    ----------
    points : array_like
        points on the road length from 0 to road length

    Returns
    -------
    array_like
        calculated initial density data
    """
    values = 1 * (points <= 5) + (-4 + points) * (points > 5) * (points <= 6) \
        + 2 * (points > 6) * (points < 15) + (2 * points - 28) \
        * (points > 15) * (points <= 16) + 4 * (points > 16) * (points < 25) \
        + 1 * (points >= 25)
    return values


def main():
    """Run the LWR model and visualize the results."""
    params = LWR_PARAMS.copy()

    # Length of road
    length = 35
    # Spacial Grid Resolution
    grid_resolution = 150
    # CFL = 0.95
    dx = length / grid_resolution
    x = np.arange(0, length, dx)
    # initial density Points
    U = initial(x)

    params["initial_conditions"] = U
    params['boundary_conditions'] = "loop"
    # params['boundary_conditions'] = "extend_both"
    # params['boundary_conditions'] = {"constant_both": (4, 2)}

    env = LWR(params)
    run(env, rl_actions=1, visualize=True)


if __name__ == "__main__":
    main()
