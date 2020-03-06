"""Evaluates the baseline performance of merge without RL control.

Baseline is no AVs.
"""

import numpy as np
from flow.core.experiment import Experiment
from flow.benchmarks.merge0 import flow_params


def merge_baseline(num_runs, render=True):
    """Run script for all merge baselines.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render: bool, optional
            specifies whether to use the gui during execution

    Returns
    -------
        flow.core.experiment.Experiment
            class needed to run simulations
    """
    sim_params = flow_params['sim']
    env_params = flow_params['env']

    # modify the rendering to match what is requested
    sim_params.render = render

    # set the evaluation flag to True
    env_params.evaluate = True

    flow_params['env'].horizon = env_params.horizon
    exp = Experiment(flow_params)

    results = exp.run(num_runs)
    avg_speed = np.mean(results['returns'])

    return avg_speed


if __name__ == '__main__':
    runs = 2  # number of simulations to average over
    res = merge_baseline(num_runs=runs, render=False)

    print('---------')
    print('The average speed across {} runs is {}'.format(runs, res))
