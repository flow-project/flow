"""Bottleneck runner script for generating flow-density plots.

Run density experiment to generate capacity diagram for the
bottleneck experiment
"""

import multiprocessing
import numpy as np
import os
import ray

from examples.sumo.bottlenecks import bottleneck_example


@ray.remote
def run_bottleneck(flow_rate, num_trials, num_steps, render=None):
    """Run a rollout of the bottleneck environment.

    Parameters
    ----------
    flow_rate : float
        bottleneck inflow rate
    num_trials : int
        number of rollouts to perform
    num_steps : int
        number of simulation steps per rollout
    render : bool
        whether to render the environment

    Returns
    -------
    float
        average outflow rate across rollouts
    float
        average speed across rollouts
    float
        average rollout density outflow
    list of float
        per rollout outflows
    float
        inflow rate
    """
    print('Running experiment for inflow rate: ', flow_rate, render)
    exp = bottleneck_example(flow_rate, num_steps, restart_instance=True)
    info_dict = exp.run(num_trials, num_steps)

    return info_dict['average_outflow'], \
        np.mean(info_dict['velocities']), \
        np.mean(info_dict['average_rollout_density_outflow']), \
        info_dict['per_rollout_outflows'], \
        flow_rate


if __name__ == '__main__':
    # import the experiment variable`
    densities = list(range(400, 3000, 100))
    outflows = []
    velocities = []
    bottleneckdensities = []

    per_step_densities = []
    per_step_avg_velocities = []
    per_step_outflows = []

    rollout_inflows = []
    rollout_outflows = []

    num_cpus = multiprocessing.cpu_count()
    ray.init(num_cpus=max(num_cpus - 2, 1))
    bottleneck_outputs = [run_bottleneck.remote(d, 10, 2000)
                          for d in densities]
    for output in ray.get(bottleneck_outputs):
        outflow, velocity, bottleneckdensity, \
            per_rollout_outflows, flow_rate = output
        for i, _ in enumerate(per_rollout_outflows):
            rollout_outflows.append(per_rollout_outflows[i])
            rollout_inflows.append(flow_rate)
        outflows.append(outflow)
        velocities.append(velocity)
        bottleneckdensities.append(bottleneckdensity)

    path = os.path.dirname(os.path.abspath(__file__))
    np.savetxt(path + '/../../data/rets.csv',
               np.matrix([densities,
                          outflows,
                          velocities,
                          bottleneckdensities]).T,
               delimiter=',')
    np.savetxt(path + '/../../data/inflows_outflows.csv',
               np.matrix([rollout_inflows,
                          rollout_outflows]).T,
               delimiter=',')
