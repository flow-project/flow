"""Evaluates the baseline performance of grid0 without RL control.

Baseline is an actuated traffic light provided by SUMO.
"""

import numpy as np
from flow.core.experiment import Experiment
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.benchmarks.grid0 import flow_params
from flow.benchmarks.grid0 import N_ROWS
from flow.benchmarks.grid0 import N_COLUMNS


def grid0_baseline(num_runs, render=True):
    """Run script for the grid0 baseline.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render : bool, optional
            specifies whether to use the gui during execution

    Returns
    -------
        flow.core.experiment.Experiment
            class needed to run simulations
    """
    exp_tag = flow_params['exp_tag']
    sim_params = flow_params['sim']
    vehicles = flow_params['veh']
    env_params = flow_params['env']
    net_params = flow_params['net']
    initial_config = flow_params.get('initial', InitialConfig())

    # define the traffic light logic
    tl_logic = TrafficLightParams(baseline=False)

    phases = [{"duration": "31", "minDur": "8", "maxDur": "45",
               "state": "GrGr"},
              {"duration": "6", "minDur": "3", "maxDur": "6",
               "state": "yryr"},
              {"duration": "31", "minDur": "8", "maxDur": "45",
               "state": "rGrG"},
              {"duration": "6", "minDur": "3", "maxDur": "6",
               "state": "ryry"}]

    for i in range(N_ROWS * N_COLUMNS):
        tl_logic.add('center'+str(i), tls_type='actuated', phases=phases,
                     programID=1)

    # modify the rendering to match what is requested
    sim_params.render = render

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
    env = env_class(env_params, sim_params, scenario)

    exp = Experiment(env)

    results = exp.run(num_runs, env_params.horizon)
    total_delay = np.mean(results['returns'])

    return total_delay


if __name__ == '__main__':
    runs = 1  # number of simulations to average over
    res = grid0_baseline(num_runs=runs)

    print('---------')
    print('The total delay across {} runs is {}'.format(runs, res))
