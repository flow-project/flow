"""Evaluates the baseline performance of grid1 without RL control.

Baseline is an actuated traffic light provided by SUMO.
"""

import numpy as np
from flow.core.experiment import Experiment
from flow.core.params import TrafficLightParams
from flow.benchmarks.grid1 import flow_params
from flow.benchmarks.grid1 import N_ROWS
from flow.benchmarks.grid1 import N_COLUMNS


def grid1_baseline(num_runs, render=True):
    """Run script for the grid1 baseline.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render: bool, optional
            specifies whether to the gui during execution

    Returns
    -------
        flow.core.experiment.Experiment
            class needed to run simulations
    """
    sim_params = flow_params['sim']
    env_params = flow_params['env']

    # define the traffic light logic
    tl_logic = TrafficLightParams(baseline=False)
    phases = [{'duration': '31', 'minDur': '5', 'maxDur': '45',
               "state": "GrGr"},
              {'duration': '2', 'minDur': '2', 'maxDur': '2',
               "state": "yryr"},
              {'duration': '31', 'minDur': '5', 'maxDur': '45',
               "state": "rGrG"},
              {'duration': '2', 'minDur': '2', 'maxDur': '2',
               "state": "ryry"}]
    for i in range(N_ROWS*N_COLUMNS):
        tl_logic.add('center'+str(i), tls_type='actuated', phases=phases,
                     programID=1)

    # modify the rendering to match what is requested
    sim_params.render = render

    # set the evaluation flag to True
    env_params.evaluate = True

    flow_params['env'].horizon = env_params.horizon
    exp = Experiment(flow_params)

    results = exp.run(num_runs)
    total_delay = np.mean(results['returns'])

    return total_delay


if __name__ == '__main__':
    runs = 1  # number of simulations to average over
    res = grid1_baseline(num_runs=runs, render=False)

    print('---------')
    print('The total delay across {} runs is {}'.format(runs, res))
