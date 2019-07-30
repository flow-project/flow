from flow.controllers import IDMController, ContinuousRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams, SumoCarFollowingParams, InFlows, \
    SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.highway_ramps import HighwayRampsScenario, \
                                         ADDITIONAL_NET_PARAMS
import os
from time import strftime

import csv
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors


for length_before in [500]:
    for length_between in [350]:
        for highway_inflow_rate in [4000]:
            for ramp_inflow_rate in [200]:
                add_net_params = ADDITIONAL_NET_PARAMS.copy()
                add_net_params['highway_length'] = 2 * length_before + length_between
                add_net_params['on_ramps_length'] = length_before / 2
                add_net_params['off_ramps_length'] = length_before / 2
                add_net_params['highway_lanes'] = 3
                add_net_params['on_ramps_lanes'] = 1
                add_net_params['off_ramps_lanes'] = 2
                add_net_params['highway_speed'] = 30
                add_net_params['on_ramps_speed'] = 20
                add_net_params['off_ramps_speed'] = 20
                add_net_params['on_ramps_pos'] = [length_before]
                add_net_params['off_ramps_pos'] = [length_before + length_between]
                add_net_params['next_off_ramp_proba'] = 0.25

                inflows = InFlows()
                inflows.add(
                    veh_type="idm_all_checks",
                    edge="highway_0",
                    vehs_per_hour=highway_inflow_rate,
                    depart_lane="free",
                    depart_speed="speedLimit",
                    name="highway_inflow")
                inflows.add(
                    veh_type="idm_safe_speed",
                    edge="on_ramp_0",
                    # vehs_per_hour=ramp_inflow_rate,
                    probability=0.1,
                    depart_lane="free",
                    depart_speed="speedLimit",
                    name="on_ramp_inflow")

                net_params = NetParams(
                    inflows=inflows,
                    additional_params=add_net_params)

                vehicles = VehicleParams()
                vehicles.add(
                    veh_id="idm_safe_speed",
                    # acceleration_controller=(IDMController, {
                    #     "noise": 0.2
                    # }),
                    car_following_params=SumoCarFollowingParams(
                        speed_mode=23,
                        tau=1.5
                    ),
                    # lane_change_params=SumoLaneChangeParams(lane_change_mode="strategic"))
                    lane_change_params=SumoLaneChangeParams(lane_change_mode=1621))
                vehicles.add(
                    veh_id="idm_all_checks",
                    # acceleration_controller=(IDMController, {
                    #     "noise": 0.2
                    # }),
                    car_following_params=SumoCarFollowingParams(
                        speed_mode="all_checks",
                        tau=1.5
                    ),
                    lane_change_params=SumoLaneChangeParams(lane_change_mode=1621))
                    # lane_change_params=SumoLaneChangeParams(lane_change_mode="strategic"))

                env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
                sim_params = SumoParams(
                    sim_step=0.2,
                    emission_path="./pi_data",
                    render=True,
                    restart_instance=True)
                initial_config = InitialConfig()

                scenario = HighwayRampsScenario(
                    name="pi",
                    vehicles=vehicles,
                    net_params=net_params,
                    initial_config=initial_config)

                env = AccelEnv(env_params, sim_params, scenario)

                exp = Experiment(env)
                exp.run(1, 8000, convert_to_csv=True)
                # exit(0)

                csv_emissions_path = os.path.join(sim_params.emission_path,
                                                "{0}-emission.csv".format(scenario.name))

                veh = defaultdict(defaultdict(list).copy)

                with open(csv_emissions_path, newline='') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        for key in ['time', 'x', 'y', 'edge_id', 'relative_position',
                                    'lane_number', 'speed']:
                            veh[row['id']][key].append(row[key])

                all_time_steps = [v['time'] for v in veh.values()]
                time_steps = [ts for ts_list in all_time_steps for ts in ts_list]
                time_steps = sorted(list(map(float, list(set(time_steps)))))

                n_veh = len(veh.keys())
                n_steps = len(time_steps)

                time2step = {}
                for i, time in enumerate(time_steps):
                    time2step[time] = i

                # n_steps x n_veh matrix specifying the absolute position of every vehicle at
                # every time step. Set to zero if the vehicle is not present in the network at
                # that time step.
                pos = np.zeros((n_steps, n_veh))

                h0_length = length_before
                h1_length = length_between

                for veh_count, veh_id in enumerate(veh):
                    for i, time in enumerate(veh[veh_id]['time']):
                        step = time2step[float(time)]
                        edge_id = veh[veh_id]['edge_id'][i]
                        rel_pos = veh[veh_id]['relative_position'][i]
                        if edge_id == 'highway_0':
                            pos[step][veh_count] = float(rel_pos)
                        elif edge_id == ':highway_1_2':
                            pos[step][veh_count] = h0_length + float(rel_pos)
                            # print(edge_id, rel_pos)
                        elif edge_id == 'highway_1':
                            pos[step][veh_count] = h0_length + 12 + float(rel_pos)
                        elif edge_id == ':highway_2_2':
                            pos[step][veh_count] = h0_length + 12 + h1_length + float(rel_pos)
                            # print(edge_id, rel_pos)
                        elif edge_id == 'highway_2':
                            pos[step][veh_count] = h0_length + 24 + h1_length + float(rel_pos)


                # n_steps x n_veh matrix specifying the speed of every vehicle at every time
                # step. Set to zero if the vehicle is not present in the network at that time
                # step.
                speed = np.zeros((n_steps, n_veh))

                for veh_count, veh_id in enumerate(veh):
                    for i, time in enumerate(veh[veh_id]['time']):
                        step = time2step[float(time)]
                        speed[step][veh_count] = float(veh[veh_id]['speed'][i])

                # (n_steps,) vector representing the unique time steps in the simulation.
                time = np.zeros((n_steps,))

                time[:] = time_steps[:]

                ###
                ### plot diagram like in time_space_diagram.py
                ###

                highway_max_speed = add_net_params['highway_speed']

                # some plotting parameters
                cdict = {
                    'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
                    'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
                    'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
                }
                my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

                # perform plotting operation
                fig = plt.figure(figsize=(16, 9))
                ax = plt.axes()
                norm = plt.Normalize(0, highway_max_speed)
                cols = []

                xmin = time[0]
                xmax = time[-1]
                xbuffer = (xmax - xmin) * 0.025  # 2.5% of range
                ymin, ymax = np.amin(pos), np.amax(pos)
                ybuffer = (ymax - ymin) * 0.025  # 2.5% of range

                ax.set_xlim(xmin - xbuffer, xmax + xbuffer)
                ax.set_ylim(ymin - ybuffer, ymax + ybuffer)

                for indx_car in range(pos.shape[1]):
                    unique_car_pos = pos[:, indx_car]

                    # discontinuity from wraparound
                    disc = np.where(np.abs(np.diff(unique_car_pos)) >= 10)[0] + 1
                    unique_car_time = np.insert(time, disc, np.nan)
                    unique_car_pos = np.insert(unique_car_pos, disc, np.nan)
                    unique_car_speed = np.insert(speed[:, indx_car], disc, np.nan)

                    points = np.array(
                        [unique_car_time, unique_car_pos]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap=my_cmap, norm=norm)

                    # Set the values used for color mapping
                    lc.set_array(unique_car_speed)
                    lc.set_linewidth(1.75)
                    cols.append(lc)

                title = str(add_net_params)
                title = title[:len(title)//2] + "\n" + title[len(title)//2:] + "\n"
                title += ("length_before={}, length_between={}, highway_inflow_rate={}, "
                        "ramp_inflow_rate={}".format(length_before, length_between,
                                                    highway_inflow_rate, ramp_inflow_rate))

                plt.title(title, fontsize=10)
                plt.ylabel('Position on highway (m)', fontsize=20)
                plt.xlabel('Time (s)', fontsize=20)

                for col in cols:
                    line = ax.add_collection(col)
                cbar = plt.colorbar(line, ax=ax)
                cbar.set_label('Velocity (m/s)', fontsize=20)
                cbar.ax.tick_params(labelsize=18)

                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                ###########################################################################
                #                      Note: For MergeScenario only                       #
                plt.plot(time, [0] * pos.shape[0], linewidth=3, color="white")        #
                plt.plot(time, [-0.1] * pos.shape[0], linewidth=3, color="white")     #
                ###########################################################################

                # plt.show()
                filename = "lengths_{}_{}_flows_{}_{}".format(
                    length_before, length_between, highway_inflow_rate, ramp_inflow_rate
                )
                timestamp = strftime('%d-%m-%Y_%H-%M-%S')

                plt.savefig('./pi_data/{}_{}.png'.format(filename, timestamp))
                plt.close()
