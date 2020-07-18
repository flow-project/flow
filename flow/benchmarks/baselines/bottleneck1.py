"""Evaluates the baseline performance of bottleneck1 without RL control.

Baseline is no AVs.
"""

import numpy as np
from flow.core.experiment import Experiment
from flow.core.params import InFlows
from flow.core.params import SumoLaneChangeParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import ContinuousRouter
from flow.benchmarks.bottleneck1 import flow_params
from flow.benchmarks.bottleneck1 import SCALING


def bottleneck1_baseline(num_runs, render=True):
    """Run script for the bottleneck1 baseline.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render: str, optional
            specifies whether to use the gui during execution

    Returns
    -------
        flow.core.experiment.Experiment
            class needed to run simulations
    """
    sim_params = flow_params['sim']
    env_params = flow_params['env']
    net_params = flow_params['net']

    # we want no autonomous vehicles in the simulation
    vehicles = VehicleParams()
    vehicles.add(veh_id='human',
                 car_following_params=SumoCarFollowingParams(
                     speed_mode=9,
                 ),
                 routing_controller=(ContinuousRouter, {}),
                 lane_change_params=SumoLaneChangeParams(
                     lane_change_mode=1621,
                 ),
                 num_vehicles=1 * SCALING)

    # only include human vehicles in inflows
    flow_rate = 2300 * SCALING
    inflow = InFlows()
    inflow.add(veh_type='human', edge='1',
               vehs_per_hour=flow_rate,
               departLane='random', departSpeed=10)
    net_params.inflows = inflow

    # modify the rendering to match what is requested
    sim_params.render = render

    # set the evaluation flag to True
    env_params.evaluate = True

    flow_params['env'].horizon = env_params.horizon
    exp = Experiment(flow_params)

    results = exp.run(num_runs)

    return np.mean(results['returns']), np.std(results['returns'])


if __name__ == '__main__':
    runs = 2  # number of simulations to average over
    mean, std = bottleneck1_baseline(num_runs=runs, render=False)

    print('---------')
    print('The average outflow, std. deviation over 500 seconds '
          'across {} runs is {}, {}'.format(runs, mean, std))
