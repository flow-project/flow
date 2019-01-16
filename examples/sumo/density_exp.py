"""
Run density experiment to generate capacity diagram for the
bottleneck experiment
"""

import multiprocessing
import numpy as np
import os
import ray

from examples.sumo.bottlenecks import bottleneck_example

STEP_SIZE = 100
NUM_TRIALS = 10


@ray.remote
def run_bottleneck(flow_rate, num_trials, num_steps, render=None):
    print('Running experiment for inflow rate: ', flow_rate, render)
    exp = bottleneck_example(flow_rate, num_steps, restart_instance=True)
    info_dict = exp.run(num_trials, num_steps)

    return info_dict['average_outflow'], \
        np.mean(info_dict['velocities']), \
        np.mean(info_dict['average_rollout_density_outflow']), \
        info_dict['per_rollout_outflows'], \
        flow_rate, info_dict['lane_4_vels']


if __name__ == '__main__':
    # import the experiment variable`
    densities = list(range(2100, 2200, STEP_SIZE))
    outflows = []
    velocities = []
    lane_4_vels = []
    bottleneckdensities = []

    per_step_densities = []
    per_step_avg_velocities = []
    per_step_outflows = []

    rollout_inflows = []
    rollout_outflows = []

    num_cpus = multiprocessing.cpu_count()
    ray.init(num_cpus=max(num_cpus - 3, 1))
    bottleneck_outputs = [run_bottleneck.remote(d, NUM_TRIALS, NUM_STEPS)
                          for d in densities]
    for output in ray.get(bottleneck_outputs):
        outflow, velocity, bottleneckdensity, \
            per_rollout_outflows, flow_rate, lane_4_vel = output
        for i, _ in enumerate(per_rollout_outflows):
            rollout_outflows.append(per_rollout_outflows[i])
            rollout_inflows.append(flow_rate)
        outflows.append(outflow)
        velocities.append(velocity)
        lane_4_vels += lane_4_vel
        bottleneckdensities.append(bottleneckdensity)

    path = os.path.dirname(os.path.abspath(__file__))
    np.savetxt(path + '/../../flow/visualize/data/rets.csv',
               np.matrix([densities,
                          outflows,
                          velocities,
                          bottleneckdensities]).T,
               delimiter=',')
    np.savetxt(path + '/../../flow/visualize/data/inflows_outflows_test.csv',
               np.matrix([rollout_inflows,
                          rollout_outflows]).T,
               delimiter=',')
    np.savetxt(path + '/../../flow/visualize/data/inflows_velocity.csv',
               np.matrix(lane_4_vels),
               delimiter=',')
