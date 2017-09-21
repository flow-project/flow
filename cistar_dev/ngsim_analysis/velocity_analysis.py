from pudb import set_trace
import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

'Change these to your desired paths to the NGSIM data!'
input_folder_80 = "/Users/eugenevinitsky/Box Sync/Research/Bayen/Data" \
           "/NGSIM-Raw/I-80-Main-Data/vehicle-trajectory-data"
# this one is for the US 101 data
input_folder_101 = "/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/NGSIM" \
            "/US-101/vehicle-trajectory-data"
input_folder_80_proto = "/Users/eugenevinitsky/Box Sync/Research" \
            "/Bayen/Data/NGSIM/I-80prototype/vehicle-trajectory-data"
output_folder = "/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/" \
            "NGSIM-Processed"


def read_processed_data_80():
    global output_folder

    pm_4 = np.loadtxt(output_folder+"/4-415-Processed.txt", delimiter=',',
                      skiprows=1)
    pm_5 = np.loadtxt(output_folder+"/5-515-Processed.txt", delimiter=',',
                      skiprows=1)
    pm_515 = np.loadtxt(output_folder+"/515-530-Processed.txt", delimiter=',',
                        skiprows=1)
    pm_230 = np.loadtxt(output_folder+"/230-Processed.txt", delimiter=',',
                        skiprows=1)
    return pm_4, pm_5, pm_515, pm_230


''' Read in data that has been processed in the format of extract_cars for I-80
PARAMETERS: NONE
OUTPUT: three sets of data corresponding to 4-4:15, 5-5:15, 5:15-5:30 '''


def read_processed_data_101():
    global output_folder

    pm_750 = np.loadtxt(output_folder+"/750-Processed.txt", delimiter=',',
                        skiprows=1)
    pm_805 = np.loadtxt(output_folder+"/805-Processed.txt", delimiter=',',
                        skiprows=1)
    pm_820 = np.loadtxt(output_folder+"/820-Processed.txt", delimiter=',',
                        skiprows=1)

    return pm_750, pm_805, pm_820


''' Takes in a dataset and appends the velocity, obtained via differentiation
as an additional column at the end. The first and last element for each car
will be zero as we are missing a boundary condition for computing
the velocity at that point
PARAMETERS: dataset: extracted and processed data from NGSIM
            output_filename: name of file to save in
            dt: size of frame in seconds
OUPUT: file containing dataset with velocities appended in furthest
right column'''


def velocity_append(dataset, output_filename, dt=.1):
    num_cols = dataset.shape[1]
    velocities = np.zeros((dataset.shape[0], 1))
    veh_index = 1

    for i in range(dataset.shape[0] - 1):
        # check that we are on the same car
        if (int(dataset[i, 0]) != int(dataset[i+1, 0])):
            veh_index = 0
            velocities[i, 0] = 0.0
        # or if we are on the first entry for that car
        elif veh_index == 1:
            velocities[i, 0] = 0.0
        # otherwise, differentiate
        else:
            velocities[i, 0] = (dataset[i + 1, 5] - dataset[i - 1, 5])/(2*dt)

        veh_index = veh_index + 1

    # last element should also be a zero
    velocities[dataset.shape[0]-1, 0] = 0
    new_dataset = dataset
    new_dataset = np.concatenate((new_dataset, velocities), axis=1)
    # save the file
    f = open(output_filename, 'w')
    for i in range(dataset.shape[0]):
        f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6},'
                '{7}, {8}, {9}, {10}, {11}, {12}\n'.
                format(int(dataset[i, 0]), int(dataset[i, 1]),
                       dataset[i, 2], dataset[i, 3], dataset[i, 4],
                       dataset[i, 5], int(dataset[i, 6]),
                       int(dataset[i, 7]), int(dataset[i, 8]),
                       dataset[i, 9], dataset[i, 10], dataset[i, 11],
                       velocities[i, 0]))
    f.close()


''' Display a histogram of the data listed in filename
PARAMETERS: file_name: self-explanatory
            u_cutoff: maximum velocity to consider in histogram
            l_cutoff: minimum velocity to consider in histogram
OUTPUT: Plot of the histogram'''


def display_hist(file_name, u_cutoff=25, l_cutoff=.1):
    dataset = np.loadtxt(file_name, delimiter=',')
    velocities = dataset[:, -1]
    velocities = velocities[np.where(velocities > l_cutoff)]
    velocities = velocities[np.where(velocities < u_cutoff)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(velocities, bins='auto', normed='True')


''' Apply a symmetric exponential moving average filter to the velocities
listed in filename as described in Kesting, Thiemann 2008.
PARAMETERS: filename
            dt: size of frame in seconds
OUTPUT: filtered list of velocities '''


def sema_filter(file_name, output_filename, dt=.1):
    dataset = np.loadtxt(file_name, delimiter=',')
    # pull out the velocities
    vel = dataset[:, dataset.shape[1]-1]
    v_filter = np.zeros((dataset.shape[0], 1))
    # keeps track of which car we are on
    step_index = 0
    # keeps track of how many frames relate to a car
    N_alpha = 0
    # smoothing window converted to indices
    delta = 1.0/dt
    veh_id = dataset[0, 0]
    while step_index < dataset.shape[0] - 1:

        while (dataset[step_index, 0] == veh_id and
                step_index + N_alpha + 1 < dataset.shape[0]):
                N_alpha = N_alpha + 1
                veh_id = dataset[step_index + N_alpha, 0]

        veh_id = int(dataset[step_index + N_alpha, 0])

        for i in range(N_alpha):
            # note that i is indexed from 1
            D = int(np.min([np.floor(3*delta), i, N_alpha - i]))
            # Compute normalization factor
            Z = 0.0
            v_smooth = 0.0
            for k in range(i-D, i+D+1):
                Z = Z + np.exp(-np.abs(i-k)/delta)
                v_smooth = (v_smooth +
                            vel[k+step_index]*np.exp(-np.abs(i-k)/delta))
            v_filter[step_index + i, 0] = v_smooth/Z

        # move our index to the next vehicle
        print step_index
        step_index = step_index + N_alpha
        N_alpha = 0

    f = open(output_filename, 'w')
    f.write('vehID, fID, locX, locY, globalX, globalY, laneID, PrecedeID, \
        followID, spaceHeadway, timeHeadway, width, vel, filterVeloc')
    for i in range(dataset.shape[0]):
        f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6},'
                '{7}, {8}, {9}, {10}, {11}, {12}, {13}\n'.
                format(int(dataset[i, 0]), int(dataset[i, 1]),
                       dataset[i, 2], dataset[i, 3], dataset[i, 4],
                       dataset[i, 5], int(dataset[i, 6]),
                       int(dataset[i, 7]), int(dataset[i, 8]),
                       dataset[i, 9], dataset[i, 10], dataset[i, 11],
                       dataset[i, 12], v_filter[i, 0]))
    f.close()

if __name__ == '__main__':
    # pm_4, pm_5, pm_515, pm_230 = read_processed_data_80()
    # velocity_append(pm_4, output_folder+'/pm_4_velocity_append.txt', .1)
    # velocity_append(pm_5, output_folder+'/pm_5_velocity_append.txt', .1)
    # velocity_append(pm_515, output_folder+'/pm_515_velocity_append.txt', .1)
    # velocity_append(pm_230, output_folder+'/pm_230_velocity_append.txt',
    #                 1./15.0)
    # pm_750, pm_805, pm_820 = read_processed_data_101()
    # velocity_append(pm_750, output_folder+'/pm_750_velocity_append.txt', .1)
    # velocity_append(pm_5, output_folder+'/pm_805_velocity_append.txt', .1)
    # velocity_append(pm_515, output_folder+'/pm_820_velocity_append.txt', .1)

    sema_filter(output_folder+'/pm_230_velocity_append.txt',
                output_folder+'/pm_230_velocity_filter.txt')
    display_hist(output_folder+'/pm_230_velocity_filter.txt')
    plt.show()
