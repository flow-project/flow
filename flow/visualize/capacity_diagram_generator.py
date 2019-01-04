"""Generates capacity diagrams for the bottleneck"""

import csv
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
import os

rc('text', usetex=True)
font = {'weight': 'bold',
        'size': 18}
rc('font', **font)

inflows = []
outflows = []
path = os.path.dirname(os.path.abspath(__file__))
with open(path + '/../../data/inflows_outflows.csv', 'rt') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        inflows.append(float(row[0]))
        outflows.append(float(row[1]))

unique_inflows = sorted(list(set(inflows)))
sorted_outflows = {inflow: [] for inflow in unique_inflows}

for inflow, outlfow in zip(inflows, outflows):
    sorted_outflows[inflow].append(outlfow)

mean_outflows = np.asarray([np.mean(sorted_outflows[inflow])
                            for inflow in unique_inflows])
min_outflows = np.asarray([np.min(sorted_outflows[inflow])
                           for inflow in unique_inflows])
max_outflows = np.asarray([np.max(sorted_outflows[inflow])
                           for inflow in unique_inflows])
std_outflows = np.asarray([np.std(sorted_outflows[inflow])
                           for inflow in unique_inflows])

plt.figure(figsize=(27, 9))

plt.plot(unique_inflows, mean_outflows, linewidth=2, color='orange')
plt.fill_between(unique_inflows, mean_outflows - std_outflows,
                 mean_outflows + std_outflows, alpha=0.25, color='orange')
plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
plt.ylabel('Outflow' + r'$ \ \frac{vehs}{hour}$')
plt.tick_params(labelsize=20)
plt.rcParams['xtick.minor.size'] = 20
plt.minorticks_on()
plt.show()
