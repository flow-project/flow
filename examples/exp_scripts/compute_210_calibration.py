"""We load the calibrated data from calibrated_values and compute how accurate it is."""
import numpy as np
import pandas as pd
import pickle as pkl
import os

if __name__ == '__main__':
    with open(os.path.abspath('../calibrated_values/info_dict.pkl'), 'rb') as file:
        data = pkl.load(file)

    calibrated_data = pd.read_csv('../calibrated_values/i210_sub_merge_area_reduced.csv')
    valid_section = calibrated_data[calibrated_data['oid'] == 8009307]
    speeds = valid_section['speed'].to_numpy() / 3.6  # (km/h to m/s)
    density = valid_section['density']
    outflow = valid_section['flow']

    dict_to_idx = {'oid': 0, 'ent': 1, 'flow': 2, 'ttime': 3,
                   'speed': 4, 'density': 5, 'lane_changes': 6, 'total_lane_changes': 7}

    errors = []
    # compute the speed errors for a given set of params
    for experiment in data:
        merge_speed = experiment['avg_merge_speed']
        # now sum it up in segments noting that the sim step is 0.8
        num_steps = int(120 / 0.8)

        step_sizes = np.arange(0, len(merge_speed), num_steps)
        # sum up all the slices
        summed_slices = np.add.reduceat(merge_speed, step_sizes) / num_steps
        # throw away the last point and the first point before the network is formed
        error = np.abs(np.mean(summed_slices[:-1] - speeds[:summed_slices.shape[0] - 1]))
        errors.append(error)
    print(errors)
