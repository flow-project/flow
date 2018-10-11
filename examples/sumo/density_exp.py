"""
Run density experiment to generate capacity diagram
"""

import ray

from examples.sumo.bottleneck import bottleneck_example

import numpy as np

@ray.remote
def run_bottleneck(flow_rate, num_trials, num_steps, render=None):
    print("Running experiment for inflow rate: ", flow_rate, render)
    exp = bottleneck_example(flow_rate, num_steps)
    info_dict = exp.run(num_trials, num_steps)
    # per_step_avg_velocities = exp.per_step_avg_velocities[:1]
    # per_step_densities = exp.per_step_densities[:1]
    # per_step_outflows = exp.per_step_outflows[:1]

    return info_dict["average_rollout_outflow"], np.mean(info_dict["velocities"]), np.mean(info_dict["average_rollout_density_outflow"])

if __name__ == "__main__":
    # import the experiment variable
    densities = list(range(800,3001,200))
    outflows = []
    velocities = []
    bottleneckdensities = []

    per_step_densities = []
    per_step_avg_velocities = []
    per_step_outflows = []


    #
    # bottleneck_outputs = [run_bottleneck(d, 5, 1500) for d in densities]
    # for output in bottleneck_outputs:

    ray.init()
    bottleneck_outputs = [run_bottleneck.remote(d, 5, 1500) for d in densities]
    for output in ray.get(bottleneck_outputs):
        outflow, velocity, bottleneckdensity = output

        outflows.append(outflow)
        velocities.append(velocity)
        bottleneckdensities.append(bottleneckdensity)

    np.savetxt("rets.csv", np.matrix([densities, outflows, velocities, bottleneckdensities]).T, delimiter=",")

