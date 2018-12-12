"""Evaluates the baseline performance of figureeight without RL control.

Baseline is human acceleration and intersection behavior.
"""

import numpy as np
from flow.core.experiment import SumoExperiment
from flow.core.params import InitialConfig
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights
from flow.controllers import IDMController
from flow.controllers import ContinuousRouter
from flow.benchmarks.figureeight0 import flow_params


def figure_eight_baseline(num_runs, render=True):
    """Run script for all figure eight baselines.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        flow_params : dict
            the flow meta-parameters describing the structure of a benchmark.
            Must be one of the figure eight flow_params
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

    # modify the rendering to match what is requested
    sumo_params.render = render

    # set the evaluation flag to True
    env_params.evaluate = True

    # we want no autonomous vehicles in the simulation
    vehicles = Vehicles()
    vehicles.add(veh_id='human',
                 acceleration_controller=(IDMController, {'noise': 0.2}),
                 routing_controller=(ContinuousRouter, {}),
                 speed_mode='no_collide',
                 num_vehicles=14)

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
    avg_speed = np.mean(results['mean_returns'])

    return avg_speed


if __name__ == '__main__':
    runs = 2  # number of simulations to average over
    res = figure_eight_baseline(num_runs=runs, flow_params=flow_params)

    print('---------')
    print('The average speed across {} runs is {}'.format(runs, res))
