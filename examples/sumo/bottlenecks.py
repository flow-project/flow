"""File demonstrating formation of congestion in bottleneck."""

from flow.controllers import ContinuousRouter
from flow.controllers import SumoLaneChangeController
from flow.core.experiment import SumoExperiment
from flow.core.params import EnvParams
from flow.core.params import InFlows
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.traffic_lights import TrafficLights
from flow.core.util import emission_to_csv
from flow.core.vehicles import Vehicles
from flow.envs.bottleneck_env import BottleneckEnv
from flow.scenarios.bottleneck import BottleneckScenario

import logging

import numpy as np
SCALING = 1
DISABLE_TB = True
# If set to False, ALINEA will control the ramp meter
DISABLE_RAMP_METER = True
INFLOW = 1800


class BottleneckDensityExperiment(SumoExperiment):

    def __init__(self, env, scenario):
        super().__init__(env, scenario)

    def run(self, num_runs, num_steps, rl_actions=None, convert_to_csv=False):
        """Runs the given scenario for num_runs and num_steps per run.

        Parameters
        ----------
        num_runs: int
            number of runs the experiment should perform
        num_steps: int
            number of steps to be performs in each run of the experiment
        rl_actions: list or numpy ndarray, optional
            actions to be performed by rl vehicles in the network (if there are
            any)
        convert_to_csv: bool
            Specifies whether to convert the emission file created by sumo into
            a csv file
        """

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
        for i in range(num_runs):
            vel = np.zeros(num_steps)
            logging.info('Iter #' + str(i))
            ret = 0
            ret_list = []
            step_outflows = []
            step_densities = []
            vehicles = self.env.vehicles
            state = self.env.reset()
            for j in range(num_steps):
                state, reward, done, _ = self.env.step(rl_actions(state))
                vel[j] = np.mean(vehicles.get_speed(vehicles.get_ids()))
                ret += reward
                ret_list.append(reward)

                step_outflow = self.env.get_bottleneck_outflow(20)
                density = self.env.get_bottleneck_density()

                step_outflows.append(step_outflow)
                step_densities.append(density)
                if done:
                    break
            rets.append(ret)
            vels.append(vel)
            mean_densities.append(sum(step_densities[100:]) /
                                  (num_steps - 100))
            outflow = self.env.get_bottleneck_outflow(10000)
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

        info_dict['average_rollout_density_outflow'] = np.mean(mean_densities)

        print('Average, std return: {}, {}'.format(
            np.mean(rets), np.std(rets)))
        print('Average, std speed: {}, {}'.format(
            np.mean(mean_vels), np.std(std_vels)))
        self.env.terminate()

        if convert_to_csv:
            # collect the location of the emission file
            dir_path = self.env.sumo_params.emission_path
            emission_filename = \
                '{0}-emission.xml'.format(self.env.scenario.name)
            emission_path = \
                '{0}/{1}'.format(dir_path, emission_filename)

            # convert the emission file into a csv
            emission_to_csv(emission_path)

        return info_dict


def bottleneck_example(flow_rate, horizon, enable_lane_changing=False,
                       render=None, restart_instance=False):
    """Perform a simulation of vehicles on a bottleneck.

    Parameters
    ----------
    flow_rate : float
        total inflow rate of vehicles into the bottlneck
    horizon : int
        time horizon
    render: bool, optional
        specifies whether to use sumo's gui during execution
    restart_instance: bool, optional
        whether to restart SUMO on every new run

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a bottleneck.
    """
    if render is None:
        render = False

    sumo_params = SumoParams(
        sim_step=0.5,
        render=render,
        overtake_right=False,
        restart_instance=restart_instance)

    vehicles = Vehicles()

    lane_change_mode = 512
    if enable_lane_changing:
        lane_change_mode = 1621

    vehicles.add(
        veh_id='human',
        speed_mode='all_checks',
        lane_change_controller=(SumoLaneChangeController, {}),
        routing_controller=(ContinuousRouter, {}),
        lane_change_mode=lane_change_mode,
        num_vehicles=1)

    additional_env_params = {
        'target_velocity': 40,
        'max_accel': 1,
        'max_decel': 1,
        'lane_change_duration': 5,
        'add_rl_if_exit': False,
        'disable_tb': DISABLE_TB,
        'disable_ramp_metering': DISABLE_RAMP_METER
    }
    env_params = EnvParams(
        horizon=horizon, additional_params=additional_env_params)

    inflow = InFlows()
    inflow.add(
        veh_type='human',
        edge='1',
        vehsPerHour=flow_rate,
        departLane='random',
        departSpeed=10)

    traffic_lights = TrafficLights()
    if not DISABLE_TB:
        traffic_lights.add(node_id='2')
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id='3')

    additional_net_params = {'scaling': SCALING}
    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing='random',
        min_gap=5,
        lanes_distribution=float('inf'),
        edges_distribution=['2', '3', '4', '5'])

    scenario = BottleneckScenario(
        name='bay_bridge_toll',
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    env = BottleneckEnv(env_params, sumo_params, scenario)

    return BottleneckDensityExperiment(env, scenario)


if __name__ == '__main__':
    # import the experiment variable
    # inflow, number of steps, binary
    exp = bottleneck_example(INFLOW, 1000, render=True)
    exp.run(5, 1000)
