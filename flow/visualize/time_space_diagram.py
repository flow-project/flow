"""Generate a time space diagram for some networks.

This method accepts as input a csv file containing the sumo-formatted emission
file, and then uses this data to generate a time-space diagram, with the x-axis
being the time (in seconds), the y-axis being the position of a vehicle, and
color representing the speed of te vehicles.

If the number of simulation steps is too dense, you can plot every nth step in
the plot by setting the input `--steps=n`.

Note: This script assumes that the provided network has only one lane on the
each edge, or one lane on the main highway in the case of MergeNetwork.

Usage
-----
::
    python time_space_diagram.py </path/to/emission>.csv </path/to/params>.json
"""
from flow.utils.rllib import get_flow_params
from flow.networks import RingNetwork, FigureEightNetwork, MergeNetwork, I210SubNetwork

import argparse
import csv
try:
    from matplotlib import pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import numpy as np


# networks that can be plotted by this method
ACCEPTABLE_NETWORKS = [
    RingNetwork,
    FigureEightNetwork,
    MergeNetwork,
    I210SubNetwork
]


def import_data_from_emission(fp):
    r"""Import relevant data from the predefined emission (.csv) file.

    Parameters
    ----------
    fp : str
        file path (for the .csv formatted file)

    Returns
    -------
    dict of dict
        Key = "veh_id": name of the vehicle \n Elements:

        * "time": time step at every sample
        * "edge": edge ID at every sample
        * "pos": relative position at every sample
        * "vel": speed at every sample
    """
    # initialize all output variables
    veh_id, t, edge, rel_pos, vel, lane = [], [], [], [], [], []

    # import relevant data from emission file
    for record in csv.DictReader(open(fp)):
        veh_id.append(record['id'])
        t.append(record['time'])
        edge.append(record['edge_id'])
        rel_pos.append(record['relative_position'])
        vel.append(record['speed'])
        lane.append(record['lane_number'])

    # we now want to separate data by vehicle ID
    ret = {key: {'time': [], 'edge': [], 'pos': [], 'vel': [], 'lane': []}
           for key in np.unique(veh_id)}
    for i in range(len(veh_id)):
        ret[veh_id[i]]['time'].append(float(t[i]))
        ret[veh_id[i]]['edge'].append(edge[i])
        ret[veh_id[i]]['pos'].append(float(rel_pos[i]))
        ret[veh_id[i]]['vel'].append(float(vel[i]))
        ret[veh_id[i]]['lane'].append(float(lane[i]))

    return ret


def get_time_space_data(data, params):
    r"""Compute the unique inflows and subsequent outflow statistics.

    Parameters
    ----------
    data : dict of dict
        Key = "veh_id": name of the vehicle \n Elements:

        * "time": time step at every sample
        * "edge": edge ID at every sample
        * "pos": relative position at every sample
        * "vel": speed at every sample
    params : dict
        flow-specific parameters, including:

        * "network" (str): name of the network that was used when generating
          the emission file. Must be one of the network names mentioned in
          ACCEPTABLE_NETWORKS,
        * "net_params" (flow.core.params.NetParams): network-specific
          parameters. This is used to collect the lengths of various network
          links.

    Returns
    -------
    as_array
        n_steps x n_veh matrix specifying the absolute position of every
        vehicle at every time step. Set to zero if the vehicle is not present
        in the network at that time step.
    as_array
        n_steps x n_veh matrix specifying the speed of every vehicle at every
        time step. Set to zero if the vehicle is not present in the network at
        that time step.
    as_array
        a (n_steps,) vector representing the unique time steps in the
        simulation

    Raises
    ------
    AssertionError
        if the specified network is not supported by this method
    """
    # check that the network is appropriate
    assert params['network'] in ACCEPTABLE_NETWORKS, \
        'Network must be one of: ' + ', '.join(ACCEPTABLE_NETWORKS)

    # switcher used to compute the positions based on the type of network
    switcher = {
        RingNetwork: _ring_road,
        MergeNetwork: _merge,
        FigureEightNetwork: _figure_eight,
        I210SubNetwork: _i210_subnetwork
    }

    # Collect a list of all the unique times.
    all_time = []
    for veh_id in data.keys():
        all_time.extend(data[veh_id]['time'])
    all_time = np.sort(np.unique(all_time))

    # Get the function from switcher dictionary
    func = switcher[params['network']]

    # Execute the function
    pos, speed, all_time = func(data, params, all_time)

    return pos, speed, all_time


def _merge(data, params, all_time):
    r"""Generate position and speed data for the merge.

    This only include vehicles on the main highway, and not on the adjacent
    on-ramp.

    Parameters
    ----------
    data : dict of dict
        Key = "veh_id": name of the vehicle \n Elements:

        * "time": time step at every sample
        * "edge": edge ID at every sample
        * "pos": relative position at every sample
        * "vel": speed at every sample
    params : dict
        flow-specific parameters
    all_time : array_like
        a (n_steps,) vector representing the unique time steps in the
        simulation

    Returns
    -------
    as_array
        n_steps x n_veh matrix specifying the absolute position of every
        vehicle at every time step. Set to zero if the vehicle is not present
        in the network at that time step.
    as_array
        n_steps x n_veh matrix specifying the speed of every vehicle at every
        time step. Set to zero if the vehicle is not present in the network at
        that time step.
    """
    # import network data from flow params
    inflow_edge_len = 100
    premerge = params['net'].additional_params['pre_merge_length']
    postmerge = params['net'].additional_params['post_merge_length']

    # generate edge starts
    edgestarts = {
        'inflow_highway': 0,
        'left': inflow_edge_len + 0.1,
        'center': inflow_edge_len + premerge + 22.6,
        'inflow_merge': inflow_edge_len + premerge + postmerge + 22.6,
        'bottom': 2 * inflow_edge_len + premerge + postmerge + 22.7,
        ':left_0': inflow_edge_len,
        ':center_0': inflow_edge_len + premerge + 0.1,
        ':center_1': inflow_edge_len + premerge + 0.1,
        ':bottom_0': 2 * inflow_edge_len + premerge + postmerge + 22.6
    }

    # compute the absolute position
    for veh_id in data.keys():
        data[veh_id]['abs_pos'] = _get_abs_pos(data[veh_id]['edge'],
                                               data[veh_id]['pos'], edgestarts)

    # prepare the speed and absolute position in a way that is compatible with
    # the space-time diagram, and compute the number of vehicles at each step
    pos = np.zeros((all_time.shape[0], len(data.keys())))
    speed = np.zeros((all_time.shape[0], len(data.keys())))
    for i, veh_id in enumerate(sorted(data.keys())):
        for spd, abs_pos, ti, edge in zip(data[veh_id]['vel'],
                                          data[veh_id]['abs_pos'],
                                          data[veh_id]['time'],
                                          data[veh_id]['edge']):
            # avoid vehicles outside the main highway
            if edge in ['inflow_merge', 'bottom', ':bottom_0']:
                continue
            ind = np.where(ti == all_time)[0]
            pos[ind, i] = abs_pos
            speed[ind, i] = spd

    return pos, speed, all_time


def _ring_road(data, params, all_time):
    r"""Generate position and speed data for the ring road.

    Vehicles that reach the top of the plot simply return to the bottom and
    continue.

    Parameters
    ----------
    data : dict of dict
        Key = "veh_id": name of the vehicle \n Elements:

        * "time": time step at every sample
        * "edge": edge ID at every sample
        * "pos": relative position at every sample
        * "vel": speed at every sample
    params : dict
        flow-specific parameters
    all_time : array_like
        a (n_steps,) vector representing the unique time steps in the
        simulation

    Returns
    -------
    as_array
        n_steps x n_veh matrix specifying the absolute position of every
        vehicle at every time step. Set to zero if the vehicle is not present
        in the network at that time step.
    as_array
        n_steps x n_veh matrix specifying the speed of every vehicle at every
        time step. Set to zero if the vehicle is not present in the network at
        that time step.
    """
    # import network data from flow params
    ring_length = params['net'].additional_params["length"]
    junction_length = 0.1  # length of inter-edge junctions

    edgestarts = {
        "bottom": 0,
        ":right_0": 0.25 * ring_length,
        "right": 0.25 * ring_length + junction_length,
        ":top_0": 0.5 * ring_length + junction_length,
        "top": 0.5 * ring_length + 2 * junction_length,
        ":left_0": 0.75 * ring_length + 2 * junction_length,
        "left": 0.75 * ring_length + 3 * junction_length,
        ":bottom_0": ring_length + 3 * junction_length
    }

    # compute the absolute position
    for veh_id in data.keys():
        data[veh_id]['abs_pos'] = _get_abs_pos(data[veh_id]['edge'],
                                               data[veh_id]['pos'], edgestarts)

    # create the output variables
    pos = np.zeros((all_time.shape[0], len(data.keys())))
    speed = np.zeros((all_time.shape[0], len(data.keys())))
    for i, veh_id in enumerate(sorted(data.keys())):
        for spd, abs_pos, ti in zip(data[veh_id]['vel'],
                                    data[veh_id]['abs_pos'],
                                    data[veh_id]['time']):
            ind = np.where(ti == all_time)[0]
            pos[ind, i] = abs_pos
            speed[ind, i] = spd

    return pos, speed, all_time


def _i210_subnetwork(data, params, all_time):
    r"""Generate position and speed data for the i210 subnetwork.

    We only look at the second to last lane of edge 119257908#1-AddedOnRampEdge

    Parameters
    ----------
    data : dict of dict
        Key = "veh_id": name of the vehicle \n Elements:

        * "time": time step at every sample
        * "edge": edge ID at every sample
        * "pos": relative position at every sample
        * "vel": speed at every sample
    params : dict
        flow-specific parameters
    all_time : array_like
        a (n_steps,) vector representing the unique time steps in the
        simulation

    Returns
    -------
    as_array
        n_steps x n_veh matrix specifying the absolute position of every
        vehicle at every time step. Set to zero if the vehicle is not present
        in the network at that time step.
    as_array
        n_steps x n_veh matrix specifying the speed of every vehicle at every
        time step. Set to zero if the vehicle is not present in the network at
        that time step.
    """
    # import network data from flow params
    #
    # edge_starts = {"119257908#0": 0,
    #                "119257908#1-AddedOnRampEdge": 686.98}
    desired_lane = 1
    edge_starts = {"119257914": 0,
                   "119257908#0": 61.58,
                   "119257908#1-AddedOnRampEdge": 686.98 + 61.58}
    # edge_starts = {"119257908#0": 0}
    # edge_starts = {"119257908#1-AddedOnRampEdge": 0}
    # desired_lane = 5

    # compute the absolute position
    for veh_id in data.keys():
        data[veh_id]['abs_pos'] = _get_abs_pos_1_edge(data[veh_id]['edge'],
                                                      data[veh_id]['pos'],
                                                      edge_starts)

    # create the output variables
    # TODO(@ev) handle subsampling better than this
    low_time = int(0 / params['sim'].sim_step)
    high_time = int(1600 / params['sim'].sim_step)
    all_time = all_time[low_time:high_time]

    # track only vehicles that were around during this time period
    observed_row_list = []
    pos = np.zeros((all_time.shape[0], len(data.keys())))
    speed = np.zeros((all_time.shape[0], len(data.keys())))
    for i, veh_id in enumerate(sorted(data.keys())):
        for spd, abs_pos, ti, edge, lane in zip(data[veh_id]['vel'],
                                                data[veh_id]['abs_pos'],
                                                data[veh_id]['time'],
                                                data[veh_id]['edge'],
                                                data[veh_id]['lane']):
            # avoid vehicles not on the relevant edges. Also only check the second to
            # last lane
            if edge not in edge_starts.keys() or ti not in all_time or lane != desired_lane:
                continue
            else:
                if i not in observed_row_list:
                    observed_row_list.append(i)
            ind = np.where(ti == all_time)[0]
            pos[ind, i] = abs_pos
            speed[ind, i] = spd

    pos = pos[:, observed_row_list]
    speed = speed[:, observed_row_list]

    return pos, speed, all_time


def _figure_eight(data, params, all_time):
    r"""Generate position and speed data for the figure eight.

    The vehicles traveling towards the intersection from one side will be
    plotted from the top downward, while the vehicles from the other side will
    be plotted from the bottom upward.

    Parameters
    ----------
    data : dict of dict
        Key = "veh_id": name of the vehicle \n Elements:

        * "time": time step at every sample
        * "edge": edge ID at every sample
        * "pos": relative position at every sample
        * "vel": speed at every sample
    params : dict
        flow-specific parameters
    all_time : array_like
        a (n_steps,) vector representing the unique time steps in the
        simulation

    Returns
    -------
    as_array
        n_steps x n_veh matrix specifying the absolute position of every
        vehicle at every time step. Set to zero if the vehicle is not present
        in the network at that time step.
    as_array
        n_steps x n_veh matrix specifying the speed of every vehicle at every
        time step. Set to zero if the vehicle is not present in the network at
        that time step.
    """
    # import network data from flow params
    net_params = params['net']
    ring_radius = net_params.additional_params['radius_ring']
    ring_edgelen = ring_radius * np.pi / 2.
    intersection = 2 * ring_radius
    junction = 2.9 + 3.3 * net_params.additional_params['lanes']
    inner = 0.28

    # generate edge starts
    edgestarts = {
        'bottom': inner,
        'top': intersection / 2 + junction + inner,
        'upper_ring': intersection + junction + 2 * inner,
        'right': intersection + 3 * ring_edgelen + junction + 3 * inner,
        'left': 1.5 * intersection + 3 * ring_edgelen + 2 * junction + 3 * inner,
        'lower_ring': 2 * intersection + 3 * ring_edgelen + 2 * junction + 4 * inner,
        ':bottom_0': 0,
        ':center_1': intersection / 2 + inner,
        ':top_0': intersection + junction + inner,
        ':right_0': intersection + 3 * ring_edgelen + junction + 2 * inner,
        ':center_0': 1.5 * intersection + 3 * ring_edgelen + junction + 3 * inner,
        ':left_0': 2 * intersection + 3 * ring_edgelen + 2 * junction + 3 * inner,
        # for aimsun
        'bottom_to_top': intersection / 2 + inner,
        'right_to_left': junction + 3 * inner,
    }

    # compute the absolute position
    for veh_id in data.keys():
        data[veh_id]['abs_pos'] = _get_abs_pos(data[veh_id]['edge'],
                                               data[veh_id]['pos'], edgestarts)

    # create the output variables
    pos = np.zeros((all_time.shape[0], len(data.keys())))
    speed = np.zeros((all_time.shape[0], len(data.keys())))
    for i, veh_id in enumerate(sorted(data.keys())):
        for spd, abs_pos, ti in zip(data[veh_id]['vel'],
                                    data[veh_id]['abs_pos'],
                                    data[veh_id]['time']):
            ind = np.where(ti == all_time)[0]
            pos[ind, i] = abs_pos
            speed[ind, i] = spd

    # reorganize data for space-time plot
    figure_eight_len = 6 * ring_edgelen + 2 * intersection + 2 * junction + 10 * inner
    intersection_loc = [edgestarts[':center_1'] + intersection / 2,
                        edgestarts[':center_0'] + intersection / 2]
    pos[pos < intersection_loc[0]] += figure_eight_len
    pos[np.logical_and(pos > intersection_loc[0], pos < intersection_loc[1])] \
        += - intersection_loc[1]
    pos[pos > intersection_loc[1]] = \
        - pos[pos > intersection_loc[1]] + figure_eight_len + intersection_loc[0]

    return pos, speed, all_time


def _get_abs_pos(edge, rel_pos, edgestarts):
    """Compute the absolute positions from edges and relative positions.

    This is the variable we will ultimately use to plot individual vehicles.

    Parameters
    ----------
    edge : list of str
        list of edges at every time step
    rel_pos : list of float
        list of relative positions at every time step
    edgestarts : dict
        the absolute starting position of every edge

    Returns
    -------
    list of float
        the absolute positive for every sample
    """
    ret = []
    for edge_i, pos_i in zip(edge, rel_pos):
        ret.append(pos_i + edgestarts[edge_i])
    return ret


def _get_abs_pos_1_edge(edges, rel_pos, edge_starts):
    """Compute the absolute positions from a subset of edges.

    This is the variable we will ultimately use to plot individual vehicles.

    Parameters
    ----------
    edges : list of str
        list of edges at every time step
    rel_pos : list of float
        list of relative positions at every time step
    edge_starts : dict
        the absolute starting position of every edge

    Returns
    -------
    list of float
        the absolute positive for every sample
    """
    ret = []
    for edge_i, pos_i in zip(edges, rel_pos):
        if edge_i in edge_starts.keys():
            ret.append(pos_i + edge_starts[edge_i])
        else:
            ret.append(-1)
    return ret


if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Generates time space diagrams for flow networks.',
        epilog='python time_space_diagram.py </path/to/emission>.csv '
               '</path/to/flow_params>.json')

    # required arguments
    parser.add_argument('emission_path', type=str,
                        help='path to the csv file.')
    parser.add_argument('flow_params', type=str,
                        help='path to the flow_params json file.')

    # optional arguments
    parser.add_argument('--steps', type=int, default=1,
                        help='rate at which steps are plotted.')
    parser.add_argument('--title', type=str, default='Time Space Diagram',
                        help='rate at which steps are plotted.')
    parser.add_argument('--max_speed', type=int, default=8,
                        help='The maximum speed in the color range.')
    parser.add_argument('--min_speed', type=int, default=0,
                        help='The minimum speed in the color range.')
    parser.add_argument('--start', type=float, default=0,
                        help='initial time (in sec) in the plot.')
    parser.add_argument('--stop', type=float, default=float('inf'),
                        help='final time (in sec) in the plot.')

    args = parser.parse_args()

    # flow_params is imported as a dictionary
    if '.json' in args.flow_params:
        flow_params = get_flow_params(args.flow_params)
    else:
        module = __import__("examples.exp_configs.non_rl", fromlist=[args.flow_params])
        flow_params = getattr(module, args.flow_params).flow_params

    # import data from the emission.csv file
    emission_data = import_data_from_emission(args.emission_path)

    # compute the position and speed for all vehicles at all times
    pos, speed, time = get_time_space_data(emission_data, flow_params)

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
    norm = plt.Normalize(args.min_speed, args.max_speed)
    cols = []

    xmin = max(time[0], args.start)
    xmax = min(time[-1], args.stop)
    xbuffer = (xmax - xmin) * 0.025  # 2.5% of range
    ymin, ymax = np.amin(pos), np.amax(pos)
    ybuffer = (ymax - ymin) * 0.025  # 2.5% of range

    ax.set_xlim(xmin - xbuffer, xmax + xbuffer)
    ax.set_ylim(ymin - ybuffer, ymax + ybuffer)

    for indx_car in range(pos.shape[1]):
        unique_car_pos = pos[:, indx_car]

        if flow_params['network'] == I210SubNetwork:
            indices = np.where(pos[:, indx_car] != 0)[0]
            unique_car_speed = speed[indices, indx_car]
            points = np.array([time[indices], pos[indices, indx_car]]).T.reshape(-1, 1, 2)
        else:

            # discontinuity from wraparound
            disc = np.where(np.abs(np.diff(unique_car_pos)) >= 10)[0] + 1
            unique_car_time = np.insert(time, disc, np.nan)
            unique_car_pos = np.insert(unique_car_pos, disc, np.nan)
            unique_car_speed = np.insert(speed[:, indx_car], disc, np.nan)
            #
            points = np.array(
                [unique_car_time, unique_car_pos]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=my_cmap, norm=norm)

        # Set the values used for color mapping
        lc.set_array(unique_car_speed)
        lc.set_linewidth(1.75)
        cols.append(lc)

    plt.title(args.title, fontsize=25)
    plt.ylabel('Position (m)', fontsize=20)
    plt.xlabel('Time (s)', fontsize=20)

    for col in cols:
        line = ax.add_collection(col)
    cbar = plt.colorbar(line, ax=ax, norm=norm)
    cbar.set_label('Velocity (m/s)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    ###########################################################################
    #                       Note: For MergeNetwork only                       #
    if flow_params['network'] == 'MergeNetwork':                              #
        plt.plot(time, [0] * pos.shape[0], linewidth=3, color="white")        #
        plt.plot(time, [-0.1] * pos.shape[0], linewidth=3, color="white")     #
    ###########################################################################

    plt.show()
