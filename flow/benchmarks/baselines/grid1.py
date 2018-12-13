"""Evaluates the baseline performance of grid1 without RL control.

Baseline is an actuated traffic light provided by SUMO.
"""

import numpy as np
from flow.core.experiment import SumoExperiment
from flow.core.params import InitialConfig
from flow.core.params import TrafficLights
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
            specifies whether to use sumo's gui during execution

    Returns
    -------
        SumoExperiment
            class needed to run simulations
    """
    exp_tag = flow_params['exp_tag']
    sumo_params = flow_params['sumo']
    vehicles = flow_params['veh']
    env_params = flow_params['env']
    net_params = flow_params['net']
    initial_config = flow_params.get('initial', InitialConfig())

    # define the traffic light logic
    tl_logic = TrafficLights(baseline=False)
    phases = [{'duration': '31', 'minDur': '5', 'maxDur': '45',
               'state': 'GGGrrrGGGrrr'},
              {'duration': '2', 'minDur': '2', 'maxDur': '2',
               'state': 'yyyrrryyyrrr'},
              {'duration': '31', 'minDur': '5', 'maxDur': '45',
               'state': 'rrrGGGrrrGGG'},
              {'duration': '2', 'minDur': '2', 'maxDur': '2',
               'state': 'rrryyyrrryyy'}]
    for i in range(N_ROWS*N_COLUMNS):
        tl_logic.add('center'+str(i), tls_type='actuated', phases=phases,
                     programID=1)

    # modify the rendering to match what is requested
    sumo_params.render = render

    # set the evaluation flag to True
    env_params.evaluate = True

    # import the scenario class
    module = __import__('flow.scenarios', fromlist=[flow_params['scenario']])
    scenario_class = getattr(module, flow_params['scenario'])

    # create the scenario object
    scenario = scenario_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic
    )

    # import the environment class
    module = __import__('flow.envs', fromlist=[flow_params['env_name']])
    env_class = getattr(module, flow_params['env_name'])

    # create the environment object
    env = env_class(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    results = exp.run(num_runs, env_params.horizon)
    total_delay = np.mean(results['returns'])

    return total_delay


if __name__ == '__main__':
    runs = 1  # number of simulations to average over
    res = grid1_baseline(num_runs=runs, render=False)

    print('---------')
    print('The total delay across {} runs is {}'.format(runs, res))
