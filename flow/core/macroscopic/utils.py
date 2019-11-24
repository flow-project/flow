"""Utility methods for the macroscopic models."""
import matplotlib.pyplot as plt
import numpy as np


class DictDescriptor(object):
    """Dictionary object with descriptor the the individual elements.

    TODO: describe
    """

    def __init__(self, *args):
        """Instantiate the object.

        Parameters
        ----------
        args : (Any, Any, str), iterable
            specifies the key, value, and description of each element in the
            dictionary
        """
        self._dict = {}
        self._descriptions = {}
        self._types = {}

        for arg in args:
            key, value, typ, description = arg

            # in case the same key was used twice, raise an AssertionError
            assert key not in self._dict.keys(), \
                "Key variable '{}' was used twice".format(key)

            # add the new values
            self._dict[key] = value
            self._descriptions[key] = description
            self._types[key] = typ

    def copy(self):
        """Return the dictionary object."""
        return self._dict.copy()

    def description(self, key):
        """Return the description of the specific element."""
        return self._descriptions.get(key, "")

    def type(self, key):
        """Return the description of the specific element."""
        return self._types.get(key, "")


def visualize_plots(x, all_densities, all_speeds, time_steps):
    """Create surface plot for density and velocity evolution of simulation.

    Parameters
    ----------
    x : array-like or list
        points of the road length to plot against
    all_densities: N x M array-like matrix
        density values on the road length M at every time step N.
    all_speeds: N x M array-like matrix
        velocity values on the road length M at every time step N.
    time_steps: list
        discrete time steps that the simulation has run for
    """
    # density plot
    fig, plots = plt.subplots(2, figsize=(10, 10))
    fig.subplots_adjust(hspace=.5)
    y_vector, x_vector = np.meshgrid(x, time_steps)
    first_plot = plots[0].contourf(x_vector, y_vector, all_densities, levels=900, cmap='jet')
    plots[0].set(ylabel='Length (Position on Street in meters)', xlabel='Time (seconds)')
    plots[0].set_title('Density Evolution')
    color_bar = fig.colorbar(first_plot, ax=plots[0], shrink=0.8)
    color_bar.ax.set_title('Density\nLevels', fontsize=8)

    # velocity plot
    second_plot = plots[1].contourf(x_vector, y_vector, all_speeds, levels=900, cmap='jet')
    plots[1].set(ylabel='Length (Position on Street in meters)', xlabel='Time (seconds)')
    plots[1].set_title('Velocity Evolution (m/s)')
    color_bar1 = fig.colorbar(second_plot, ax=plots[1], shrink=0.8)
    color_bar1.ax.set_title('Velocity\nLevels (m/s)', fontsize=8)
    plt.show()


def run(env, rl_actions=1, visualize=True):
    """Execute a rollout of the model.

    Parameters
    ----------
    env : flow.core.macroscopic.base_model.MacroModelEnv
        the environment to evaluate over
    rl_actions : float or list of float
        the actions to be performed by RL agents
    visualize : bool
        whether to plot the results once the rollout is done
    """
    # initialize of simulation initial values
    obs = env.reset()

    # initialize plotting values
    t = 0
    time_steps = [t]
    all_densities = [obs[:int(obs.shape[0]/2)]]
    all_speeds = [obs[int(obs.shape[0]/2):]]
    x = np.arange(0, env.length, env.dx)

    while t < env.total_time:
        # run simulation step
        obs, rew, done, _ = env.step(rl_actions)
        t = t + env.dt

        if visualize:
            # store values
            time_steps.append(t)
            density = obs[:int(obs.shape[0]/2)]
            speeds = obs[int(obs.shape[0]/2):]
            all_densities = np.concatenate((all_densities, [density]), axis=0)
            all_speeds = np.concatenate((all_speeds, [speeds]), axis=0)

    if visualize:
        # call visualize function
        visualize_plots(x, all_densities, all_speeds, time_steps)
