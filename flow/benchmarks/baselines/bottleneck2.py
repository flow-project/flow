"""Evaluates the baseline performance of bottleneck2 without RL control.

Baseline is no AVs.
"""

import numpy as np
from flow.core.experiment import SumoExperiment
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import SumoLaneChangeParams
from flow.core.params import SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights
from flow.controllers import ContinuousRouter
from flow.benchmarks.bottleneck2 import flow_params
from flow.benchmarks.bottleneck2 import SCALING


def bottleneck2_baseline(num_runs, render=True):
    """Run script for the bottleneck2 baseline.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render : bool, optional
            specifies whether to use sumo's gui during execution

    Returns
    -------
        SumoExperiment
            class needed to run simulations
    """
    exp_tag = flow_params['exp_tag']
    sumo_params = flow_params['sumo']
    env_params = flow_params['env']
    net_params = flow_params['net']
    initial_config = flow_params.get('initial', InitialConfig())
    traffic_lights = flow_params.get('tls', TrafficLights())

    # we want no autonomous vehicles in the simulation
    vehicles = Vehicles()
    vehicles.add(veh_id='human',
                 sumo_car_following_params=SumoCarFollowingParams(
                     speed_mode=9,
                 ),
                 routing_controller=(ContinuousRouter, {}),
                 sumo_lc_params=SumoLaneChangeParams(
                     lane_change_mode=0,
                 ),
                 num_vehicles=1 * SCALING)

    # only include human vehicles in inflows
    flow_rate = 1900 * SCALING
    inflow = InFlows()
    inflow.add(veh_type='human', edge='1',
               vehs_per_hour=flow_rate,
               departLane='random', departSpeed=10)
    net_params.inflows = inflow

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
        traffic_lights=traffic_lights
    )

    # import the environment class
    module = __import__('flow.envs', fromlist=[flow_params['env_name']])
    env_class = getattr(module, flow_params['env_name'])

    # create the environment object
    env = env_class(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    results = exp.run(num_runs, env_params.horizon)

    return np.mean(results['returns']), np.std(results['returns'])


if __name__ == '__main__':
    runs = 2  # number of simulations to average over
    mean, std = bottleneck2_baseline(num_runs=runs, render=False)

    print('---------')
    print('The average outflow, std. deviation over 500 seconds '
          'across {} runs is {}, {}'.format(runs, mean, std))
