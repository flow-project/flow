"""Generates capacity diagrams for the bottleneck"""

import argparse
import csv
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
import os

COLOR_LIST = ['blue', 'red', 'green']

rc('text', usetex=True)
font = {'weight': 'bold',
        'size': 18}
rc('font', **font)

plt.figure(figsize=(27, 9))
for i, file_name in enumerate(['inflows_outflows.csv', 'bottleneck_outflow_SA.txt',
                               'bottleneck_outflow_MA.txt']):
    inflows = []
    outflows = []
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/data/' + file_name, 'rt') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            inflows.append(float(row[0]))
            outflows.append(float(row[1]))

    unique_inflows = sorted(list(set(inflows)))
    sorted_outflows = {inflow: [] for inflow in unique_inflows}

    for inflow, outflow in zip(inflows, outflows):
        sorted_outflows[inflow].append(outflow)

    mean_outflows = np.asarray([np.mean(sorted_outflows[inflow])
                                for inflow in unique_inflows])
    min_outflows = np.asarray([np.min(sorted_outflows[inflow])
                               for inflow in unique_inflows])
    max_outflows = np.asarray([np.max(sorted_outflows[inflow])
                               for inflow in unique_inflows])
    std_outflows = np.asarray([np.std(sorted_outflows[inflow])
                               for inflow in unique_inflows])

    plt.plot(unique_inflows, mean_outflows, linewidth=2, color=COLOR_LIST[i])
    # plt.fill_between(unique_inflows, mean_outflows - std_outflows,
    #                  mean_outflows + std_outflows, alpha=0.25, color=COLOR_LIST[i])
    # plt.fill_between(unique_inflows, min_outflows,
    #                  max_outflows, alpha=0.1, color=COLOR_LIST[i])
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Outflow' + r'$ \ \frac{vehs}{hour}$')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    plt.minorticks_on()
    plt.title('Inflow vs. Outflow for Lane Changes Disabled')
    plt.legend(['Uncontrolled', 'RL-centralized', 'RL-decentralized'])

plt.figure(figsize=(27, 9))
for i, file_name in enumerate(['inflows_outflows_LC.csv', 'bottleneck_outflow_SA_LC.txt',
                               'bottleneck_outflow_MA_LC_LSTM.txt']):
    inflows = []
    outflows = []
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/data/' + file_name, 'rt') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            inflows.append(float(row[0]))
            outflows.append(float(row[1]))

    unique_inflows = sorted(list(set(inflows)))
    sorted_outflows = {inflow: [] for inflow in unique_inflows}

    for inflow, outflow in zip(inflows, outflows):
        sorted_outflows[inflow].append(outflow)

    mean_outflows = np.asarray([np.mean(sorted_outflows[inflow])
                                for inflow in unique_inflows])
    min_outflows = np.asarray([np.min(sorted_outflows[inflow])
                               for inflow in unique_inflows])
    max_outflows = np.asarray([np.max(sorted_outflows[inflow])
                               for inflow in unique_inflows])
    std_outflows = np.asarray([np.std(sorted_outflows[inflow])
                               for inflow in unique_inflows])

    plt.plot(unique_inflows, mean_outflows, linewidth=2, color=COLOR_LIST[i])
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Outflow' + r'$ \ \frac{vehs}{hour}$')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    #plt.minorticks_on()
    plt.title('Inflow vs. Outflow for Lane Changes Enabled')
    plt.legend(['Uncontrolled', 'RL-centralized', 'RL-decentralized'])

plt.show()
