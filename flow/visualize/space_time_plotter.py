"""
Plotting utility for ring roads.
Currently configured to take in a CSV of emission data
for a two-lane ring road experiment and return a space-time
plot with vehicle traces color-coded by speed.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import pandas as pd
from itertools import groupby

def emission_space_time_plot(data, title = 'Space-time Plot', filename="space_time_plot", save=False, show=True):
    cdict = {'red'  :  ((0., 0., 0.), (0.2, 1., 1.), (0.6, 1., 1.), (1., 0., 0.)),
         'green':  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 1., 1.), (1., 1., 1.)),
         'blue' :  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 0., 0.), (1., 0., 0.))}

    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    fig, ax = plt.subplots(2, 1, sharex = True, figsize=(16, 9))
    norm = plt.Normalize(0, round(max(data['speed']))+1)

    LENGTH = 230
    edgelen = LENGTH/4
    edgestarts = {"bottom": 0, "right": edgelen, "top": 2 * edgelen, "left": 3 * edgelen}
    start = lambda edge: edgestarts[edge]

    data['pos'] = np.array(data['relative_position']) + np.array(list(map(start, data['edge_id'])))

    cols0 = []
    cols1 = []
    for car in np.unique(data['id']):
        cardata = data[data['id'] == car]
        pos = np.array(cardata['pos'])
        vel = np.array(cardata['speed'])
        time = np.array(cardata['time'])
        lane = np.array(cardata['lane_number']).astype(float)

        disc_cond = np.logical_or(np.abs(np.diff(pos)) >= 5, abs(np.diff(lane)) > 0.01)
        disc = np.where(disc_cond)[0] + 1

        time = np.insert(time, disc, np.nan)
        pos = np.insert(pos, disc, np.nan)
        vel = np.insert(vel, disc, np.nan)
        lane = np.insert(lane, disc, np.nan)

        split_time = [list(v) for k,v in groupby(time,np.isfinite) if k]
        split_pos = [list(v) for k,v in groupby(pos,np.isfinite) if k]
        split_vel = [list(v) for k,v in groupby(vel,np.isfinite) if k]
        split_lane = [list(v) for k,v in groupby(lane,np.isfinite) if k]
        assert(len(split_pos) == len(split_vel))
        assert(len(split_pos) == len(split_time))
        assert(len(split_time) == len(split_vel))

        for i in range(len(split_pos)):
            this_time = split_time[i]
            this_pos = split_pos[i]
            this_vel = split_vel[i]
            this_lane = split_lane[i]

            points = np.array([this_time, this_pos]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=my_cmap, norm=norm)
            # Set the values used for colormapping
            lc.set_array(np.array(this_vel))
            if car[:2] == 'rl':
                lc.set_linewidth(5)
            else:
                lc.set_linewidth(1.75)
            if np.allclose(this_lane, 0):
                cols0.append(lc)
            if np.allclose(this_lane, 1):
                cols1.append(lc)


    xmin, xmax = min(time), max(time)
    xbuffer = (xmax - xmin) * 0.025 # 2.5% of range
    ymin, ymax = np.amin(data['pos']), np.amax(data['pos'])
    ybuffer = (ymax - ymin) * 0.025 # 2.5% of range


    for axis in ax:
        axis.set_xlim(xmin - xbuffer, xmax + xbuffer)
        axis.set_ylim(ymin - ybuffer, ymax + ybuffer)

    fig.suptitle(title, fontsize = 24)
    ax[0].set_title("Lane 0", fontsize=18)
    ax[0].set_ylabel('Position (m)', fontsize=18)
    ax[1].set_title("Lane 1", fontsize=18)
    ax[1].set_ylabel('Position (m)', fontsize=18)
    plt.xlabel('Time (s)', fontsize=18)

    for col in cols0: line = ax[0].add_collection(col)
    cbar = plt.colorbar(line, ax = ax[0])
    cbar.set_label('Velocity (m/s)', fontsize = 18)

    for col in cols1: line = ax[1].add_collection(col)
    cbar = plt.colorbar(line, ax = ax[1])
    cbar.set_label('Velocity (m/s)', fontsize = 18)

    if save:
        plt.savefig(filename + '.png', dpi = 200)
    if show:
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv('/Users/nishant/Development/research/frankencistar/rllabcistar/test_time_rollout/two-lane-stabilization-rl.csv')
    emission_space_time_plot(data)

