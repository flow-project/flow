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
from collections import defaultdict
try:
    from matplotlib import pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import numpy as np
import pandas as pd


# networks that can be plotted by this method
ACCEPTABLE_NETWORKS = [
    RingNetwork,
    FigureEightNetwork,
    MergeNetwork,
    I210SubNetwork
]


def import_data_from_trajectory(fp, params=dict()):
    r"""Import and preprocess data from the Flow trajectory (.csv) file.
    Parameters
    ----------
    fp : str
        file path (for the .csv formatted file)
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
    pd.DataFrame
    """
    # Read trajectory csv into pandas dataframe
    df = pd.read_csv(fp)

    # Convert column names for backwards compatibility using emissions csv
    column_conversions = {
        'time': 'time_step',
        'lane_number': 'lane_id',
    }
    df = df.rename(columns=column_conversions)
    if 'distance' not in df.columns:
        df['distance'] = _get_abs_pos(df, params)

    # Compute line segment ends by shifting dataframe by 1 row
    df[['next_pos', 'next_time']] = df.groupby('id')[['distance', 'time_step']].shift(-1)

    # Remove nans from data
    df = df[df['next_time'].notna()]

    return df


def get_time_space_data(data, params):
    r"""Compute the unique inflows and subsequent outflow statistics.
    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data
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
    ndarray (or dict of ndarray)
        3d array (n_segments x 2 x 2) containing segments to be plotted.
        every inner 2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
        in the case of I210, the nested arrays are wrapped into a dict,
        keyed on the lane number, so that each lane can be plotted
        separately.
    Raises
    ------
    AssertionError
        if the specified network is not supported by this method
    """
    # check that the network is appropriate
    assert params['network'] in ACCEPTABLE_NETWORKS, \
        'Network must be one of: ' + ', '.join([network.__name__ for network in ACCEPTABLE_NETWORKS])

    # switcher used to compute the positions based on the type of network
    switcher = {
        RingNetwork: _ring_road,
        MergeNetwork: _merge,
        FigureEightNetwork: _figure_eight,
        I210SubNetwork: _i210_subnetwork
    }

    # Get the function from switcher dictionary
    func = switcher[params['network']]

    # Execute the function
    segs, data = func(data)

    return segs, data


def _merge(data):
    r"""Generate position and speed data for the merge.
    This only include vehicles on the main highway, and not on the adjacent
    on-ramp.
    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data
    Returns
    -------
    ndarray
        3d array (n_segments x 2 x 2) containing segments to be plotted.
        every inner 2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
    pd.DataFrame
        modified trajectory dataframe
    """
    # Omit ghost edges
    keep_edges = {'inflow_merge', 'bottom', ':bottom_0'}
    data = data[data['edge_id'].isin(keep_edges)]

    segs = data[['time_step', 'distance', 'next_time', 'next_pos']].values.reshape((len(data), 2, 2))

    return segs, data


def _ring_road(data):
    r"""Generate position and speed data for the ring road.
    Vehicles that reach the top of the plot simply return to the bottom and
    continue.
    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data
    Returns
    -------
    ndarray
        3d array (n_segments x 2 x 2) containing segments to be plotted.
        every inner 2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
    pd.DataFrame
        unmodified trajectory dataframe
    """
    segs = data[['time_step', 'distance', 'next_time', 'next_pos']].values.reshape((len(data), 2, 2))

    return segs, data


def _i210_subnetwork(data):
    r"""Generate time and position data for the i210 subnetwork.
    We generate plots for all lanes, so the segments are wrapped in
    a dictionary.
    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data
    Returns
    -------
    dict of ndarray
        dictionary of 3d array (n_segments x 2 x 2) containing segments
        to be plotted. the dictionary is keyed on lane numbers, with the
        values being the 3d array representing the segments. every inner
        2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
    pd.DataFrame
        modified trajectory dataframe
    """
    # Omit ghost edges
    omit_edges = {'ghost0', '119257908#3'}
    data.loc[:, :] = data[~data['edge_id'].isin(omit_edges)]

    # Reset lane numbers that are offset by ramp lanes
    offset_edges = set(data[data['lane_id'] == 5]['edge_id'].unique())
    data.loc[data['edge_id'].isin(offset_edges), 'lane_id'] -= 1

    segs = dict()
    for lane, df in data.groupby('lane_id'):
        segs[lane] = df[['time_step', 'distance', 'next_time', 'next_pos']].values.reshape((len(df), 2, 2))

    return segs, data


def _figure_eight(data):
    r"""Generate position and speed data for the figure eight.
    The vehicles traveling towards the intersection from one side will be
    plotted from the top downward, while the vehicles from the other side will
    be plotted from the bottom upward.
    Parameters
    ----------
    data : pd.DataFrame
        cleaned dataframe of the trajectory data
    Returns
    -------
    ndarray
        3d array (n_segments x 2 x 2) containing segments to be plotted.
        every inner 2d array is comprised of two 1d arrays representing
        [start time, start distance] and [end time, end distance] pairs.
    pd.DataFrame
        unmodified trajectory dataframe
    """
    segs = data[['time_step', 'distance', 'next_time', 'next_pos']].values.reshape((len(data), 2, 2))

    return segs, data


def _get_abs_pos(df, params):
    """Compute the absolute positions from edges and relative positions.
    This is the variable we will ultimately use to plot individual vehicles.
    Parameters
    ----------
    df : pd.DataFrame
        dataframe of trajectory data
    params : dict
        flow-specific parameters
    Returns
    -------
    pd.Series
        the absolute positive for every sample
    """
    if params['network'] == MergeNetwork:
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
    elif params['network'] == RingNetwork:
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
    elif params['network'] == FigureEightNetwork:
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
    else:
        edgestarts = defaultdict(float)

    ret = df.apply(lambda x: x['relative_position'] + edgestarts[x['edge_id']], axis=1)

    if params['network'] == FigureEightNetwork:
        # reorganize data for space-time plot
        figure_eight_len = 6 * ring_edgelen + 2 * intersection + 2 * junction + 10 * inner
        intersection_loc = [edgestarts[':center_1'] + intersection / 2,
                            edgestarts[':center_0'] + intersection / 2]
        ret.loc[ret < intersection_loc[0]] += figure_eight_len
        ret.loc[(ret > intersection_loc[0]) & (ret < intersection_loc[1])] += -intersection_loc[1]
        ret.loc[ret > intersection_loc[1]] = \
            - ret.loc[ret > intersection_loc[1]] + figure_eight_len + intersection_loc[0]
    return ret


def plot_tsd(ax, df, segs, args, lane=None):
    """Plot the time-space diagram.
    Take the pre-processed segments and other meta-data, then plot all the line segments.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        figure axes that will be plotted on
    df : pd.DataFrame
        data used for axes bounds and speed coloring
    segs : list of list of lists
        line segments to be plotted, where each segment is a list of two [x,y] pairs
    args : dict
        parsed arguments
    lane : int, optional
        lane number to be shown in plot title
    Returns
    -------
    None
    """
    norm = plt.Normalize(args.min_speed, args.max_speed)

    xmin = max(df['time_step'].min(), args.start)
    xmax = min(df['time_step'].max(), args.stop)
    xbuffer = (xmax - xmin) * 0.025  # 2.5% of range
    ymin, ymax = df['distance'].min(), df['distance'].max()
    ybuffer = (ymax - ymin) * 0.025  # 2.5% of range

    ax.set_xlim(xmin - xbuffer, xmax + xbuffer)
    ax.set_ylim(ymin - ybuffer, ymax + ybuffer)

    lc = LineCollection(segs, cmap=my_cmap, norm=norm)
    lc.set_array(df['speed'].values)
    lc.set_linewidth(1)
    ax.add_collection(lc)
    ax.autoscale()

    if lane:
        ax.set_title('Time-Space Diagram: Lane {}'.format(lane), fontsize=25)
    else:
        ax.set_title('Time-Space Diagram', fontsize=25)
    ax.set_ylabel('Position (m)', fontsize=20)
    ax.set_xlabel('Time (s)', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    cbar = plt.colorbar(lc, ax=ax, norm=norm)
    cbar.set_label('Velocity (m/s)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)


if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Generates time space diagrams for flow networks.',
        epilog='python time_space_diagram.py </path/to/emission>.csv '
               '</path/to/flow_params>.json')

    # required arguments
    parser.add_argument('trajectory_path', type=str,
                        help='path to the Flow trajectory csv file.')
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

    # some plotting parameters
    cdict = {
        'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
        'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
        'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
    }
    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    # Read trajectory csv into pandas dataframe
    traj_df = import_data_from_trajectory(args.trajectory_path, flow_params)

    # Convert df data into segments for plotting
    segs, traj_df = get_time_space_data(traj_df, flow_params)

    if flow_params['network'] == I210SubNetwork:
        nlanes = traj_df['lane_id'].nunique()
        fig = plt.figure(figsize=(16, 9*nlanes))

        for lane, df in traj_df.groupby('lane_id'):
            ax = plt.subplot(nlanes, 1, lane+1)

            plot_tsd(ax, df, segs[lane], args, lane)
    else:
        # perform plotting operation
        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes()

        plot_tsd(ax, traj_df, segs, args)

    ###########################################################################
    #                       Note: For MergeNetwork only                       #
    if flow_params['network'] == 'MergeNetwork':                              #
        plt.plot([df['time_step'].min(), df['time_step'].max()],
                 [0, 0], linewidth=3, color="white")        #
        plt.plot([df['time_step'].min(), df['time_step'].max()],
                 [-0.1, -0.1], linewidth=3, color="white")     #
    ###########################################################################

    plt.show()