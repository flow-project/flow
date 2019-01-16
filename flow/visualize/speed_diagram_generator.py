"""Generates capacity diagrams for the bottleneck"""

import argparse
import csv
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
args = parser.parse_args()

rc('text', usetex=True)
font = {'weight': 'bold',
        'size': 18}
rc('font', **font)

file_list = args.files

plt.figure(figsize=(27, 9))
for file_name in file_list:
    inflows = []
    vel_0_lane = []
    vel_1_lane = []
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/data/' + file_name, 'rt') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            inflows.append(float(row[0]))
            vel_0_lane.append(float(row[1]))
            vel_1_lane.append(float(row[2]))

    unique_inflows = sorted(list(set(inflows)))
    sorted_vels = {inflow: [] for inflow in unique_inflows}

    for inflow, vel_0, vel_1 in zip(inflows, vel_0_lane, vel_1_lane):
        sorted_vels[inflow] += [vel_0, vel_1]
    mean_vels = np.asarray([np.mean(sorted_vels[inflow])
                            for inflow in unique_inflows])
    std_vels = np.asarray([np.std(sorted_vels[inflow])
                           for inflow in unique_inflows])

    plt.plot(unique_inflows, mean_vels, linewidth=2, color='orange')
    # plt.fill_between(unique_inflows, mean_vels - std_vels,
    #                  mean_vels + std_vels, alpha=0.25, color='orange')
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Velocity' + r'$ \ \frac{m}{s}$')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    plt.minorticks_on()
plt.show()
