"""Bottleneck runner script for generating flow-density plots.

Run density experiment to generate capacity diagram for the
bottleneck experiment
"""

import argparse
import multiprocessing
import numpy as np
import os
import ray

from examples.sumo.bottlenecks import bottleneck_example


@ray.remote
def run_bottleneck(flow_rate, num_trials, num_steps, render=None, disable_ramp_meter=True, n_crit=8,
                   feedback_coef=20, lc_on=False):
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
    exp = bottleneck_example(flow_rate, num_steps, render=render, restart_instance=True,
                             disable_ramp_meter=disable_ramp_meter,
                             feedback_coef=feedback_coef, n_crit=n_crit, lc_on=lc_on)
    info_dict = exp.run(num_trials, num_steps)

    return info_dict['average_outflow'], \
        np.mean(info_dict['velocities']), \
        np.mean(info_dict['average_rollout_density_outflow']), \
        info_dict['per_rollout_outflows'], \
        flow_rate, info_dict['lane_4_vels']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Runs the bottleneck exps and stores the results for processing')
    parser.add_argument('--render', action='store_true', help='Display the scenarios')

    parser.add_argument('--ramp_meter', action='store_true', help='If set, ALINEA is active in this scenario')
    parser.add_argument('--alinea_sweep', action='store_true', help='If set, perform a hyperparam sweep over ALINEA '
                                                                    'hyperparams')
    parser.add_argument('--inflow_min', type=int, default=400)
    parser.add_argument('--inflow_max', type=int, default=2500)
    parser.add_argument('--ncrit_min', type=int, default=6)
    parser.add_argument('--ncrit_max', type=int, default=12)
    parser.add_argument('--ncrit_step_size', type=int, default=1)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--num_trials', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=2000)
    parser.add_argument('--lc_on', action='store_true')
    parser.add_argument('--clear_data', action='store_true', help='If true, clean the folder where the files are '
                                                                  'stored before running anything')
    args = parser.parse_args()

    path = os.path.dirname(os.path.abspath(__file__))
    outer_path = '../../flow/visualize/trb_data/human_driving'

    if args.clear_data:
        for the_file in os.listdir(os.path.join(path, outer_path)):
            file_path = os.path.join(os.path.join(path, outer_path), the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    n_crit_range = list(range(args.ncrit_min, args.ncrit_max + args.ncrit_step_size, args.ncrit_step_size))
    feedback_coef_range = [5, 10, 20, 40, 100]

    # import the experiment variable`
    densities = list(range(args.inflow_min, args.inflow_max + args.step_size, args.step_size))

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
    ray.init(num_cpus=max(num_cpus - 4, 1))
    if args.alinea_sweep:
        for n_crit in n_crit_range:
            for feedback_coef in feedback_coef_range:

                outflows = []
                velocities = []
                lane_4_vels = []
                bottleneckdensities = []

                per_step_densities = []
                per_step_avg_velocities = []
                per_step_outflows = []

                rollout_inflows = []
                rollout_outflows = []
                bottleneck_outputs = [run_bottleneck.remote(d, args.num_trials, args.horizon, render=args.render,
                                                            disable_ramp_meter=not args.ramp_meter,
                                                            lc_on=args.lc_on,
                                                            feedback_coef=feedback_coef, n_crit=n_crit)
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

                # save the returns
                if args.lc_on:
                    ret_string = 'rets_LC_n{}_fcoeff{}_alinea.csv'.format(n_crit, feedback_coef)
                    inflow_outflow_str = 'inflows_outflows_LC_n{}_fcoeff{}_alinea.csv'.format(n_crit, feedback_coef)
                    inflow_velocity_str = 'inflows_velocity_LC_n{}_fcoeff{}_alinea.csv'.format(n_crit, feedback_coef)

                else:
                    ret_string = 'rets_n{}_fcoeff{}_alinea.csv'.format(n_crit, feedback_coef)
                    inflow_outflow_str = 'inflows_outflows_n{}_fcoeff{}_alinea.csv'.format(n_crit, feedback_coef)
                    inflow_velocity_str = 'inflows_velocity_n{}_fcoeff{}_alinea.csv'.format(n_crit, feedback_coef)

                ret_path = os.path.join(path, os.path.join(outer_path, ret_string))
                outflow_path = os.path.join(path, os.path.join(outer_path, inflow_outflow_str))
                vel_path = os.path.join(path, os.path.join(outer_path, inflow_velocity_str))

                with open(ret_path, 'ab') as file:
                    np.savetxt(file, np.matrix([densities, outflows, velocities, bottleneckdensities]).T, delimiter=',')
                with open(outflow_path, 'ab') as file:
                    np.savetxt(file,  np.matrix([rollout_inflows, rollout_outflows]).T, delimiter=',')
                with open(vel_path, 'ab') as file:
                    np.savetxt(file,  np.matrix(lane_4_vels), delimiter=',')


    else:
        bottleneck_outputs = [run_bottleneck.remote(d, args.num_trials, args.horizon, render=args.render,
                                                    lc_on=args.lc_on)
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
        np.savetxt(path + '/../../flow/visualize/trb_data/human_driving/rets_LC.csv',
                   np.matrix([densities,
                              outflows,
                              velocities,
                              bottleneckdensities]).T,
                   delimiter=',')
        np.savetxt(path + '/../../flow/visualize/trb_data/human_driving/inflows_outflows_LC.csv',
                   np.matrix([rollout_inflows,
                              rollout_outflows]).T,
                   delimiter=',')
        np.savetxt(path + '/../../flow/visualize/trb_data/human_driving/inflows_velocity_LC.csv',
                   np.matrix(lane_4_vels),
                   delimiter=',')
