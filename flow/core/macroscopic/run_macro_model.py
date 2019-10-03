r"""Runner script for RLlib experiments with macroscopic models.

To run an experiment with this script, type:

    python run_macro_model.py MODEL_NAME --{simulate, train}

where, model name is one of: {"ARZ", "LWR", "NonLocal"}. Furthermore, the
"simulate" and "train" arguments specify whether to run a training operation or
simply a simulation of a model given specific parameters. More in the next
upcoming sections.


Running Simulations
-------------------

Simulations of a specific model WITHOUT training can be performed by including
the `--simulate` flag. This looks something like the following:

    python run_macro_model.py MODEL_NAME --simulate

Simulation-based arguments that can be passed to this script include:

  * n_itr (int): number of rollouts performed by the script
  * plot_results (stores True): whether to plot and speed and density values as
    a function of time once the simulations are done
  * save_results (stores True): whether to save the speed and density plots
    before exiting
  * save_path (str): path to save the results. If not define, the images will
    be saved in flow/core/macroscopic/results/YYYY-MM-DD-hh:mm:ss.


Visualizing Trained Results
---------------------------

When running a generic simulation, the actions passed to the model are set to
None, meaning that, in the case of speed limit control environments, the speed
limit is not modified. If you have previously trained a policy using RLlib (see
the next section) and would like to visualize the performance of the policy on
a model with a similar observation space, you can use the `--checkpoint_path`
and `checkpoint_num` argument as follows:

    python run_macro_model.py MODEL_NAME --simulate \
        --checkpoint_path /path/to/results/folder \
        --checkpoint_num INT

These arguments are passed similar to result_dir and checkpoint_num in
flow/visualize/visualizer_rllib.py. If you would further like the model
parameters be collected from the checkpoint path, include the
`--include_params` flag as follows:

    python run_macro_model.py MODEL_NAME --simulate \
        --checkpoint_path /path/to/results/folder \
        --checkpoint_num INT \
        --include_params

Note that this final parameter is only valid when visualizing pre-trained
results, and will add any additional arguments that are passed (see the Passing
Additional Arguments section).


Running Training Operations
---------------------------

Simulations of a specific model WITHOUT training can be performed by including
the `--simulate` flag. This looks something like the following:

    python run_macro_model.py MODEL_NAME --train

Training-based arguments that can be passed to this script include:

  * n_itr (int): number of training epochs before the operation is exited
  * n_rollouts (int): number of rollouts per training iteration
  * n_cpus (int): number CPUs to distribute experiments over. Defaults to 1.
  * seed (int): sets the seed for numpy, tensorflow, and random


Passing Additional Arguments
----------------------------

The network parameters for each model can also be passed as additional
arguments through the command line. For example, when using the LWR model, the
following is valid:

    python run_macro_model.py "LWR" --{simulate, train} --length 10000 --dx 100

Variables that are not of type float, bool, int, or str cannot be passed in
this manner, but may be modified in the original script
"""
import os
import sys
import numpy as np
import argparse
from time import strftime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata
import json

import gym
from gym.envs.registration import register

from flow.core.util import ensure_dir
from flow.utils.rllib import get_rllib_config

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

cdict = {'red': ((0., 0., 0.), (0.2, 1., 1.), (0.6, 1., 1.), (1., 0., 0.)),
         'green': ((0., 0., 0.), (0.2, 0., 0.), (0.6, 1., 1.), (1., 1., 1.)),
         'blue': ((0., 0., 0.), (0.2, 0., 0.), (0.6, 0., 0.), (1., 0., 0.))}

my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)


def run_training(n_itr,
                 n_rollouts,
                 alg="PPO",
                 alg_params=None,
                 n_cpus=1,
                 seed=None,
                 env_name="LWR",
                 env_params=None):
    """Perform a training operation.

    Parameters
    ----------
    n_itr : int
        number of training epochs before the operation is exited
    n_rollouts : int
        number of rollouts per training iteration
    alg : str
        name of the RLlib algorithm
    alg_params : dict or None
        algorithm specific features
    n_cpus : int
        number of workers to you during training
    seed : int or None
        sets the seed for numpy, tensorflow, and random. If set to None, a
        random seed is chosen
    env_name : str
        name of the model/environment. Must be one of: {"ARZ", "LWR",
        "NonLocal"}
    env_params : dict or None
        environment-specific features. See the definition of the separate
        models for more.
    """
    # import the PARAMS object from the given model
    module = __import__("flow.core.macroscopic.{}".format(env_name.lower()),
                        fromlist=['PARAMS'])
    params = getattr(module, 'PARAMS').copy()

    # if no environment parameters are passed, use the default values
    if env_params is not None:
        params.update(env_params)

    env_params = params.copy()

    # compute the horizon
    horizon = int(env_params['total_time'] / env_params['dt'])

    # collect the algorithm class from RLlib
    agent_cls = get_agent_class(alg)

    # default algorithm parameters (assuming you are using PPO)
    if alg_params is None:
        alg_params = agent_cls._default_config.copy()
        alg_params['train_batch_size'] = horizon * n_rollouts
        alg_params['gamma'] = 0.999  # discount rate
        alg_params['model'].update({'fcnet_hiddens': [32, 32]})
        alg_params['use_gae'] = True
        alg_params['lambda'] = 0.97
        alg_params['kl_target'] = 0.02
        alg_params['num_sgd_iter'] = 10
        alg_params['clip_actions'] = False  # FIXME(ev) temporary ray bug
        alg_params['horizon'] = horizon

    # update a few common algorithm parameters
    alg_params['num_workers'] = n_cpus
    alg_params['seed'] = seed

    # save the environment parameters for replay
    alg_params['env_config']['env_name'] = env_name
    alg_params['env_config']['env_params'] = env_params
    alg_params['env_config']['run'] = alg

    # register as a gym environment
    gym_name = "{}-v0".format(env_name)

    def create_env(*_):
        register(id=gym_name,
                 entry_point='flow.core.macroscopic:{}'.format(env_name),
                 kwargs={"params": env_params})
        return gym.envs.make(gym_name)

    # Register as rllib env
    register_env(gym_name, create_env)

    # initialize a ray instance
    ray.init(num_cpus=n_cpus + 1)

    _ = run_experiments({
        gym_name: {
            'run': alg,
            'env': gym_name,
            'config': {
                **alg_params
            },
            'checkpoint_freq': 20,
            "checkpoint_at_end": True,
            'max_failures': 999,
            'stop': {
                'training_iteration': n_itr,
            },
        }
    })


def rollout(env,
            agent,
            n_itr=1,
            plot_results=False,
            save_results=False,
            save_path=''):
    """Perform a series of rollouts.

    Parameters
    ----------
    env : flow.core.macroscopic.MacroModelEnv
        the environment that will be simulated
    agent : Any
        the RL agent that is used to perform actions. If set to None, no
        actions are specified
    n_itr : int
        number of simulations to be performed
    plot_results : bool
        whether to plot and speed and density values as a function of time once
        the simulations are done
    save_results : bool
        whether to save the speed and density plots before exiting
    save_path : str
        path to save the results. If not define, the images will be saved in
        flow/core/macro/results/.
    """
    v_max = env.v_max
    rho_max = env.rho_max
    all_speeds = []
    all_densities = []

    for _ in range(n_itr):
        # reset the environment
        obs = env.reset()
        densities = np.array([obs[:int(len(obs)/2)] * rho_max])
        speeds = np.array([obs[int(len(obs)/2):] * v_max])

        for _ in range(env.horizon):
            # compute the next action
            action = agent.compute(obs) if agent is not None else None

            # perform a step
            obs, rew, done, _ = env.step(action)

            # collect all relevant data for plotting
            densities = np.concatenate(
                (densities, [obs[:int(len(obs)/2)] * rho_max]), axis=0)
            speeds = np.concatenate(
                (speeds, [obs[int(len(obs)/2):] * v_max]), axis=0)

        # add to the list across all rollouts
        all_densities.append(densities)
        all_speeds.append(speeds)

    # perform (optionally) plotting operations
    if plot_results or save_results:
        # add the default save path is none was specified
        if save_path is None:
            save_path = os.path.join(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'results/'
                ),
                strftime("%Y-%m-%d-%H:%M:%S")
            )

        # ensure that the directory exists
        if save_results:
            ensure_dir(save_path)

        for indx, (rho, vel) in enumerate(zip(all_densities, all_speeds)):
            # bins of length and time
            x = np.array([np.repeat(i * env.dx, vel.shape[0])
                          for i in range(vel.shape[1])]).flatten()
            t = np.array([np.arange(0, env.dt * vel.shape[0], env.dt)
                          for _ in range(vel.shape[1])]).flatten()

            points = np.stack((x, t)).T
            grid_x, grid_y = np.mgrid[0:env.length:env.dx,
                                      0:env.total_time:env.dt]

            # process the data so that it is ready to be plotted
            vel = vel.flatten()
            rho = rho.flatten()

            vel_gridded = griddata(points, vel, (grid_x, grid_y))
            rho_gridded = griddata(points, rho, (grid_x, grid_y))

            # create a plot for the next data
            plot_speed_or_density(
                vel_gridded, env.v_max, "Speed Plot",
                plot_results, save_results,
                env=env,
                save_path=os.path.join(save_path, "speed_{}".format(indx)),
            )
            plot_speed_or_density(
                rho_gridded, env.rho_max, "Density Plot",
                plot_results, save_results,
                env=env,
                save_path=os.path.join(save_path, "density_{}".format(indx)),
            )


def plot_speed_or_density(gridded_data,
                          max_val,
                          title,
                          plot_results,
                          save_results,
                          env,
                          save_path):
    """Perform the plotting operation for the rollout method.

    Parameters
    ----------
    gridded_data : np.ndarray
        (x,y,z) data representing the position, time, and intensity of separate
        points. Intensity in this case is either speed for density
    max_val : float
        the clipping term for the colormap on the plot
    title : str
        a title for the plot
    plot_results : bool
        whether to plot and speed and density values as a function of time once
        the simulations are done
    save_results : bool
        whether to save the speed and density plots before exiting
    env : flow.core.macroscopic.MacroModelEnv
        the environment that is being plotted
    save_path : str
        path to save the results. If not define, the images will be saved in
        flow/core/macro/results/.
    """
    plt.figure(figsize=(16, 9))
    norm = plt.Normalize(0, max_val)
    plt.title(title, fontsize=25)
    plt.xlabel("time (s)", fontsize=20)
    plt.ylabel("position (m)", fontsize=20)
    plt.imshow(gridded_data, extent=(0, env.total_time, 0, env.length),
               origin='lower', aspect='auto', cmap=my_cmap, norm=norm)
    cbar = plt.colorbar()
    # FIXME: this is hacky
    if "speed" in save_path:
        cbar.set_label('speed (m/s)', fontsize=20)
    elif "density" in save_path:
        cbar.set_label('density (veh/m)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if plot_results:
        plt.show()

    if save_results:
        plt.savefig(save_path)


def load_model_env(model_name,
                   checkpoint_path=None,
                   checkpoint_num=None,
                   include_params=False,
                   model_params=None):
    """Load a model/environment from an RLlib checkpoint.

    Parameters
    ----------
    model_name : str
        The model that should be trained/simulated. Must be one of {"LWR",
        "ARZ", "NonLocal}.
    checkpoint_path : str
        the path to the RLlib results directory that is meant to be visualized
    checkpoint_num : int
        checkpoint number
    include_params : bool
        whether to collect the environment parameters from the checkpoint path.
        If not, the parameters can be passed an arguments, or will adopt their
        default values.
    model_params : dict, iterable
        additional model-specific arguments

    Returns
    -------
    flow.core.macroscopic.MacroModelEnv
        the environment that will be simulated
    Any
        the RL agent that is used to perform actions. If set to None, no
        actions are specified
    """
    # import the environment class and the default environment parameters
    module = __import__("flow.core.macroscopic.{}".format(model_name.lower()),
                        fromlist=[model_name, 'PARAMS'])
    env_class = getattr(module, model_name)

    # import the default environment parameters
    params = getattr(module, 'PARAMS').copy()

    # update the parameters to match the inputs
    if model_params is not None:

        # load data from json file
        args_list = sys.argv[1:]
        if "--additional_params" in args_list:
            file_directory = sys.argv[1:][args_list.index("--additional_params") + 1]
            json_params = json.load(open(file_directory, 'r'))
            model_params['initial_conditions'] = np.array(json_params['initial_conditions'])
            model_params['boundary_conditions'] = np.array(json_params['boundary_conditions'])

        params.update(model_params)

    if checkpoint_path is None:
        # we assume there is no agent in this case
        agent = None
    else:
        # get the RLlib configuration files
        config = get_rllib_config(checkpoint_path)
        agent_cls = get_agent_class(config['env_config']['run'])

        # collect the environment parameters from the results directory
        if include_params:
            params = config['env_config']['env_params']

        # create the agent that will be used to compute the actions
        agent = agent_cls(env=model_name, config=config)
        checkpoint = checkpoint_path + '/checkpoint_{}'.format(checkpoint_num)
        checkpoint = checkpoint + '/checkpoint-{}'.format(checkpoint_num)
        agent.restore(checkpoint)

    # create the environment
    env = env_class(params)

    return env, agent


def parse_args(args):
    """Parse arguments that are common among all operations within this script.

    Parameters
    ----------
    args : Any, iterable
        see the top-level docstring for a description of all arguments

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # required arguments
    parser.add_argument(
        'model_name', type=str,
        help='The model that should be trained/simulated. Must be one of '
             '{"LWR", "ARZ", "NonLocal}.'
    )

    # arguments for simulation operations
    parser.add_argument(
        '--simulate', action='store_true',
        help='If called, the script is used in simulation mode.'
    )
    parser.add_argument(
        '--plot_results', action='store_true',
        help='whether to plot and speed and density values as a function of '
             'time once the simulations are done'
    )
    parser.add_argument(
        '--save_results', action='store_true',
        help='whether to save the speed and density plots before exiting'
    )
    parser.add_argument(
        '--save_path', type=str, default=None,
        help='path to save the results. If not define, the images will be '
             'saved in flow/core/macro/results/.'
    )
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory containing results'
    )
    parser.add_argument(
        '--checkpoint_num', type=str,
        help='Checkpoint number.'
    )
    parser.add_argument(
        '--include_params', action='store_true',
        help='whether to collect the model input parameters from the RLlib '
             'directory\'s stored files'
    )

    # arguments for training operations
    parser.add_argument(
        '--train', action='store_true',
        help='If called, the script is used in training mode.'
    )
    parser.add_argument(
        '--n_rollouts', type=int, default=1,
        help='number of rollouts per training iteration. Defaults to 1.'
    )
    parser.add_argument(
        '--n_cpus', type=int, default=1,
        help='number CPUs to distribute experiments over. Defaults to 1.'
    )
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Sets the seed for numpy, tensorflow, and random.'
    )

    # shared arguments between training and simulation
    parser.add_argument(
        '--n_itr', type=int, default=1,
        help='In the case of simulation mode, this is the number of rollouts '
             'performed by the script. In the case of training mode, this is '
             'the number of training epochs before the operation is exited. '
             'Defaults to 1.'
    )

    # parse the arguments
    return parser.parse_known_args(args)[0]


def parse_model_args(args, model_name):
    """Parse arguments that are common among all operations within this script.

    Parameters
    ----------
    args : Any, iterable
        see the PARAMS descriptor of the argument to see which arguments can be
        passed
    model_name : str
        name of the model whose arguments can be passed. Assumed to be located
        in flow.core.macroscopic.*

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # import the PARAMS object from the given model
    module = __import__("flow.core.macroscopic.{}".format(model_name.lower()),
                        fromlist=['PARAMS'])
    params = getattr(module, 'PARAMS')
    dict_params = params.copy()

    # fill the parser with all relevant arguments
    for key in dict_params.keys():
        if params.type(key) not in [bool, int, float, str]:
            # if the type cannot be passed, skip
            continue
        elif params.type(key) == bool:
            parser.add_argument(
                '--{}'.format(key),
                action='store_true',
                help=params.description(key)
            )
        else:
            parser.add_argument(
                '--{}'.format(key),
                type=params.type(key),
                default=dict_params[key],
                help=params.description(key)
            )

    # parse the arguments
    return parser.parse_known_args(args)[0]


def main():
    """Run the main operation. See description above."""
    # parse the passed arguments
    flags = parse_args(sys.argv[1:])
    model_flags = parse_model_args(sys.argv[1:], flags.model_name)

    # perform some assertions
    assert flags.simulate or flags.train, \
        "Please pass one of the --simulate or --train arguments."
    assert not(flags.simulate and flags.train), \
        "Cannot pass both --simulate and --train simultaneously."

    if flags.simulate:
        # create the environment
        env, agent = load_model_env(
            flags.model_name,
            checkpoint_path=flags.checkpoint_path,
            checkpoint_num=flags.checkpoint_num,
            include_params=flags.include_params,
            model_params=vars(model_flags)
        )

        # perform the simulations and (optionally) save results
        rollout(
            env, agent,
            n_itr=flags.n_itr,
            plot_results=flags.plot_results,
            save_results=flags.save_results,
            save_path=flags.save_path
        )

    elif flags.train:
        run_training(
            n_itr=flags.n_itr,
            n_rollouts=flags.n_rollouts,
            alg="PPO",  # TODO: parameter?
            alg_params=None,  # TODO: parameter?
            n_cpus=flags.n_cpus,
            seed=flags.seed,
            env_name="LWR",
            env_params=vars(model_flags)
        )


if __name__ == "__main__":
    main()
