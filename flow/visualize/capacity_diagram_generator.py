"""Generates capacity diagrams for the bottleneck.

This method accepts as input a csv file containing the inflows and outflows
from several simulations as created by the file `examples/sumo/density_exp.py`,
e.g.

    1000, 978
    1000, 773
    1500, 1134
    ...

And then uses this data to generate a capacity diagram, with the x-axis being
the inflow rates and the y-axis is the outflow rate.

Usage
-----
::
    python capacity_diagram_generator.py </path/to/file>.csv
"""
import argparse
import csv
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
import os

COLOR_LIST = ['blue', 'red', 'green', 'orange', 'purple']

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
args = parser.parse_args()

# rc('text', usetex=True)
# font = {'weight': 'bold',
#         'size': 18}
# rc('font', **font)

file_list = args.files

plt.figure(figsize=(14, 10))
for i, file_name in enumerate(file_list):
    inflows = []
    outflows = []
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/data/' + file_name, 'rt') as csvfile:


def import_data_from_csv(fp):
    r"""Import inflow/outflow data from the predefined csv file.

    Parameters
    ----------
    fp : string
        file path

    Returns
    -------
    dict
        "inflows": list of all the inflows \n
        "outflows" list of the outflows matching the inflow at the same index
    """
    inflows = []
    outflows = []
    with open(fp, 'rt') as csvfile:
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
    return {'inflows': inflows, 'outflows': outflows}


def get_capacity_data(data):
    r"""Compute the unique inflows and subsequent outflow statistics.

    Parameters
    ----------
    data : dict
        "inflows": list of all the inflows \n
        "outflows" list of the outflows matching the inflow at the same index

    Returns
    -------
    as_array
        unique inflows
    as_array
        mean outflow at given inflow
    as_array
        std deviation of outflow at given inflow
    """
    unique_vals = sorted(list(set(data['inflows'])))
    sorted_outflows = {inflow: [] for inflow in unique_vals}

    for inflow, outlfow in zip(data['inflows'], data['outflows']):
        sorted_outflows[inflow].append(outlfow)

    mean = np.asarray([np.mean(sorted_outflows[val]) for val in unique_vals])
    std = np.asarray([np.std(sorted_outflows[val]) for val in unique_vals])

    return unique_vals, mean, std


def create_parser():
    """Create an argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Generates capacity diagrams for the bottleneck.',
        epilog="python capacity_diagram_generator.py </path/to/file>.csv")

    parser.add_argument('file', type=str, help='path to the csv file.')

    return parser


if __name__ == '__main__':
    # import parser arguments
    parser = create_parser()
    args = parser.parse_args()

    # import the csv file
    data = import_data_from_csv(args.file)

    # compute the mean and std of the outflows for all unique inflows
    unique_inflows, mean_outflows, std_outflows = get_capacity_data(data)

    # some plotting parameters
    rc('text', usetex=True)
    font = {'weight': 'bold', 'size': 18}
    rc('font', **font)

    # perform plotting operation
    plt.figure(figsize=(27, 9))
    plt.plot(unique_inflows, mean_outflows, linewidth=2, color='orange')
    plt.fill_between(unique_inflows, mean_outflows - std_outflows,
                     mean_outflows + std_outflows, alpha=0.25, color='orange')
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Outflow' + r'$ \ \frac{vehs}{hour}$')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    plt.minorticks_on()
    plt.legend(['Average outflow', 'Std. deviation', 'Max-min'])

    plt.legend(args.files)
    plt.show()
