import csv
import os

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

font = {'size': 18}
rc('font', **font)

def plot_file(file_name, color, marker=None, use_min_max=False):
    inflows = []
    outflows = []
    path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path,file_name), 'rt') as csvfile:
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
    if use_min_max:
        min_outflows = np.asarray([np.min(sorted_outflows[inflow])
                                   for inflow in unique_inflows])
        max_outflows = np.asarray([np.max(sorted_outflows[inflow])
                                   for inflow in unique_inflows])
        std_outflows = np.asarray([np.std(sorted_outflows[inflow])
                                   for inflow in unique_inflows])
        plt.fill_between(unique_inflows, mean_outflows - std_outflows,
                         mean_outflows + std_outflows, alpha=0.25, color=color)
        plt.fill_between(unique_inflows, min_outflows,
                         max_outflows, alpha=0.1, color=color)

    if marker:
        plt.plot(unique_inflows, mean_outflows, linewidth=2, color=color, marker=marker)
    else:
        plt.plot(unique_inflows, mean_outflows, linewidth=2, color=color)


if __name__ == '__main__':
    # First we build the inflows outflows curve
    plt.figure(figsize=(14, 10))
    plot_file('data/inflows_outflows.csv', 'b', use_min_max=True)
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Outflow' + r'$ \ \frac{vehs}{hour}$')
    plt.title('Inflow vs. Outflow for uncontrolled intersection, no LC')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    plt.minorticks_on()
    plt.legend(['Average outflow', 'Std. deviation', 'Max-min'])
    plt.legend()
    plt.show()