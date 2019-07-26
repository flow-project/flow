"""File demonstrating formation of congestion in bottleneck."""

import argparse
import logging

import numpy as np

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

from flow.scenarios.bottleneck import BottleneckScenario
from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.envs.bottleneck_env import BottleneckEnv
from flow.core.experiment import Experiment


class BottleneckDensityExperiment(Experiment):
    """Experiment object for bottleneck-specific simulations.

    Extends flow.core.experiment.Experiment
    """
    def __init__(self, env, inflow=2300):
        """Instantiate the bottleneck experiment."""
        super().__init__(env)
        self.inflow = inflow

    def run(self, num_runs, num_steps, end_len=500, rl_actions=None, convert_to_csv=False):
        """See parent class."""
        info_dict = {}
        if rl_actions is None:

            def rl_actions(*_):
                return None

        rets = []
        mean_rets = []
        ret_lists = []
        vels = []
        mean_vels = []
        std_vels = []
        mean_densities = []
        mean_outflows = []
        lane_4_vels = []
        for i in range(num_runs):
            vel = np.zeros(num_steps)
            logging.info('Iter #' + str(i))
            ret = 0
            ret_list = []
            step_outflows = []
            step_densities = []
            state = self.env.reset()
            for j in range(num_steps):
                state, reward, done, _ = self.env.step(rl_actions(state))
                vel[j] = np.mean(self.env.k.vehicle.get_speed(
                    self.env.k.vehicle.get_ids()))
                if j >= num_steps - end_len:
                    vehicles = self.env.k.vehicle
                    vehs_on_four = vehicles.get_ids_by_edge('4')
                    lanes = vehicles.get_lane(vehs_on_four)
                    lane_dict = {veh_id: lane for veh_id, lane in
                                 zip(vehs_on_four, lanes)}
                    sort_by_lane = sorted(vehs_on_four,
                                          key=lambda x: lane_dict[x])
                    num_zeros = lanes.count(0)
                    if num_zeros > 0:
                        speed_on_zero = np.mean(vehicles.get_speed(
                            sort_by_lane[0:num_zeros]))
                    else:
                        speed_on_zero = 0.0
                    if num_zeros < len(vehs_on_four):
                        speed_on_one = np.mean(vehicles.get_speed(
                            sort_by_lane[num_zeros:]))
                    else:
                        speed_on_one = 0.0
                    lane_4_vels.append([self.inflow, speed_on_zero,
                                        speed_on_one])
                ret += reward
                ret_list.append(reward)

                env = self.env
                step_outflow = env.k.vehicle.get_outflow_rate(20)
                density = self.env.get_bottleneck_density()

                step_outflows.append(step_outflow)
                step_densities.append(density)
                if done:
                    break
            rets.append(ret)
            vels.append(vel)
            mean_densities.append(sum(step_densities[100:]) /
                                  (num_steps - 100))
            env = self.env
            outflow = env.k.vehicle.get_outflow_rate(end_len)
            mean_outflows.append(outflow)
            mean_rets.append(np.mean(ret_list))
            ret_lists.append(ret_list)
            mean_vels.append(np.mean(vel))
            std_vels.append(np.std(vel))
            print('Round {0}, return: {1}'.format(i, ret))

        info_dict['returns'] = rets
        info_dict['velocities'] = vels
        info_dict['mean_returns'] = mean_rets
        info_dict['per_step_returns'] = ret_lists
        info_dict['average_outflow'] = np.mean(mean_outflows)
        info_dict['per_rollout_outflows'] = mean_outflows
        info_dict['lane_4_vels'] = lane_4_vels

        info_dict['average_rollout_density_outflow'] = np.mean(mean_densities)

        print('Average, std return: {}, {}'.format(
            np.mean(rets), np.std(rets)))
        print('Average, std speed: {}, {}'.format(
            np.mean(mean_vels), np.std(std_vels)))
        self.env.terminate()

        return info_dict


def bottleneck_example(flow_rate, horizon, restart_instance=False,
                       render=False, scaling=1, disable_ramp_meter=True, disable_tb=True,
                       lc_on=False, n_crit=8.0, q_max=1100, q_min=275, feedback_coef=20):
    """
    Perform a simulation of vehicles on a bottleneck.

    Parameters
    ----------
    flow_rate : float
        total inflow rate of vehicles into the bottleneck
    horizon : int
        time horizon
    restart_instance: bool, optional
        whether to restart the instance upon reset
    render: bool, optional
        specifies whether to use the gui during execution
    scaling: int, optional
        This sets the number of lanes so that they go from 4 * scaling -> 2 * scaling -> 1 * scaling
    disable_tb: bool, optional
        whether the toll booth should be active
    disable_ramp_meter: bool, optional
        specifies if ALINEA should be active. For more details, look at the BottleneckEnv documentation
    lc_on: bool, optional
        if true, the vehicles have LC mode 1621 which is all safe lane changes allowed. Otherwise, it is 0 for
        no lane changing.
    n_crit: float, optional
        number of vehicles in the bottleneck we feedback around. Look at BottleneckEnv for details
    q_max: float, optional
        maximum permissible ALINEA flow. Look at BottleneckEnv for details
    q_min: float, optional
        minimum permissible ALINEA flow. Look at BottleneckEnv for details
    feedback_coeff: float, optional
        gain coefficient for ALINEA. Look at BottleneckEnv for details

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a bottleneck.
    """
    if render is None:
        render = False

    sim_params = SumoParams(
        sim_step=0.5,
        render=render,
        restart_instance=restart_instance)
    vehicles = VehicleParams()

    if lc_on:
        lc_mode = 1621
    else:
        lc_mode = 0
    vehicles.add(
        veh_id="human",
        lane_change_controller=(SimLaneChangeController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=31,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=lc_mode
        ),
        num_vehicles=1)

    additional_env_params = {
        "target_velocity": 40,
        "max_accel": 3,
        "max_decel": 3,
        "lane_change_duration": 5,
        "add_rl_if_exit": False,
        "disable_tb": disable_tb,
        "disable_ramp_metering": disable_ramp_meter,
        "n_crit": n_crit,
        "q_max": q_max,
        "q_min": q_min,
        "feedback_coeff": feedback_coef
    }
    env_params = EnvParams(
        horizon=horizon, additional_params=additional_env_params)

    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="1",
        vehsPerHour=flow_rate,
        departLane="random",
        departSpeed=10)

    traffic_lights = TrafficLightParams()
    if not disable_tb:
        traffic_lights.add(node_id="2")
    if not disable_ramp_meter:
        traffic_lights.add(node_id="3")

    additional_net_params = {"scaling": scaling, "speed_limit": 23}
    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing="random",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"])

    scenario = BottleneckScenario(
        name="bay_bridge_toll",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    env = BottleneckEnv(env_params, sim_params, scenario)

    return BottleneckDensityExperiment(env, int(flow_rate))


if __name__ == '__main__':
    # import the experiment variable
    # inflow, number of steps, binary
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Runs the bottleneck exps')
    parser.add_argument('--inflow', type=int, default=2300, help='inflow value for running the experiment')
    parser.add_argument('--ramp_meter', action='store_true', help='If set, ALINEA is active in this scenario')
    args = parser.parse_args()
    exp = bottleneck_example(args.inflow, 1000, disable_ramp_meter=not args.ramp_meter, render=True)
    exp.run(5, 1000)
