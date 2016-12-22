import numpy as np
import matplotlib.pyplot as plt

'Change these to your desired paths to the NGSIM data!'
input_folder = "/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/NGSIM-Raw" \
            "/I-80-Main-Data/vehicle-trajectory-data"
output_folder = "/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/" \
            "NGSIM-Processed"

'''Outputs the NGSIM data for each of the three times.
OUTPUT: pm_4: an array corresponding to all the data from 4pm
        pm_5: an array corresponding to all the data from 5pm
        pm_515: an array corresponding to all the data from 5:15 pm
        The rows of each array are:
        vehicle id, frame_id, lane_id, local_y, mean_speed,
        mean_accel, vehicle_length, vehicle_class, followed_id,
        leader_id
'''


def read_data():
    global input_folder

    pm_4 = np.loadtxt(input_folder +
                      "/0400pm-0415pm/trajectories-0400-0415.txt")
    pm_5 = np.loadtxt(input_folder +
                      "/0500pm-0515pm/trajectories-0500-0515.txt")
    pm_515 = np.loadtxt(input_folder +
                        "/0515pm-0530pm/trajectories-0515-0530.txt")

    return pm_4, pm_5, pm_515

''' Read in data that has been processed in the format of extract_cars
PARAMETERS: NONE
OUTPUT: three sets of data corresponding to 4-4:15, 5-5:15, 5:15-5:30 '''


def read_processed_data():
    global output_folder

    pm_4 = np.loadtxt(output_folder+"/4-415-Processed.txt", delimiter=',',
                      skiprows=1)
    pm_5 = np.loadtxt(output_folder+"/5-515-Processed.txt", delimiter=',',
                      skiprows=1)
    pm_515 = np.loadtxt(output_folder+"/515-530-Processed.txt", delimiter=',',
                        skiprows=1)

    return pm_4, pm_5, pm_515

''' Outputs a text file containing only the cars, so no motorcycles
or trucks. We also only extract the left three lanes
PARAMETERS: data_set - The data_set we will process
            file_name - filename we want as our output
Output file has as its columns:
1. Vehicle ID, 2. Frame ID (1/10 of a second)
3. Local X coordinate, 4. Local Y coordinate
5. Global X, 6. Global Y, 7. Lane ID
8. Preceding Vehicle, 9. Following vehicle
10. NGSIM computed space headway, 11. NGSIM Computed time headway'''


def extract_cars(data_set, file_name):
    global output_folder
    f = open(output_folder + file_name, 'w')
    f.write('vehID, fID, locX, locY, globalX, globalY, laneID, PrecedeID, '
            'followID, spaceHeadway, timeHeadway\n')
    print data_set.shape[0]
    for i in range(data_set.shape[0]):
        lane_id = data_set[i, 13]
        # check that we are in the left 3 lanes and that we have a car
        if ((data_set[i, 10] == 2) and
                (lane_id == 1 or lane_id == 2 or lane_id == 3)):

            f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6},'
                    '{7}, {8}, {9}, {10}\n'.format(
                int(data_set[i, 0]), int(data_set[i, 1]), data_set[i, 4],
                data_set[i, 5], data_set[i, 6],
                data_set[i, 7], int(data_set[i, 13]), int(data_set[i, 14]),
                int(data_set[i, 15]), data_set[i, 16], data_set[i, 17]))
    f.close()


def num_lane_changes(data_set):
    num_lane_changes = 0

    'initialize the system'
    veh_id = int(data_set[0, 0])
    lane_id = int(data_set[0, 6])
    for i in range(1, data_set.shape[0]):
        'if we are still on the same vehicle and the lane has changed'
        if int(data_set[i, 0]) == veh_id and lane_id != int(data_set[i, 6]):
            num_lane_changes = num_lane_changes + 1
            print ('vehicle is {0}, lane was {1}, lane is {2} \n'
                   .format(veh_id, lane_id, int(data_set[i, 6])))
        veh_id = int(data_set[i, 0])
        lane_id = int(data_set[i, 6])

    return num_lane_changes

'''We go to the point at which the lane change is
identified in the data, go back 3, 4, 5 seconds
and compute the headway from the follow car at those points.
We then plot all three sets of headways. The choice
of number is based upon the statistic that the
average time of lane change is 4 +/- 2.31 from
Estimating Acceleration and Lane-Changing
Dynamics from Next Generation Simulation
Trajectory Data
PARAMETERS: dataset - the processed dataset from which we extract data
OUTPUT: Three histograms of headways at time of initiated lane change
'''


def informal_headway_extraction(data_set):
    h3 = []
    h4 = []
    h5 = []
    'initialize the system'
    veh_id = int(data_set[0, 0])
    lane_id = int(data_set[0, 6])

    'matrix of vehicle id and frame'
    'Here we are computing the probability of a car'
    'lane changing out based on the headway behind it'
    search_mat = data_set[:, 0:2]
    for i in range(1, data_set.shape[0]):
        # if we are still on the same vehicle and it has switched lanes
        # check that we haven't aborted the lane change within five secs
        # check makes sure the lane change has lasted at least five s

        if (int(data_set[i, 0]) == veh_id and
                lane_id != int(data_set[i, 6]) and
                lane_id != int(data_set[i+50, 6]) and
                veh_id == int(data_set[i+50, 0]) and
                int(data_set[i, 6]) != int(data_set[i-50, 6])):

            print ('vehicle is {0}, lane was {1}, lane is {2}'
                   .format(veh_id, lane_id, int(data_set[i, 6])))
            print 'the following vehicle is {0}'.format(int(data_set[i, 8]))
            # if there is an observed following vehicle in the old lane
            if int(data_set[i-1, 8]) != 0:

                # find the frame at which the car is actually in the new lane
                lane_change_frame = int(data_set[i, 1])
                print 'lane change frame is {0}'.format(lane_change_frame)

                # id of the car behind it since we are
                # concerned about its probability of observing a lane change
                follow_id = int(data_set[i-1, 8])

                # frame of the leading car when it "initiated" the lane change
                frame_min3 = int(lane_change_frame) - 30
                frame_min4 = int(lane_change_frame) - 40
                frame_min5 = int(lane_change_frame) - 50
                # find the vehicle and frame corresponding to this frame
                for j in range(1, data_set.shape[0]):

                    # we've found the following car
                    if int(search_mat[j, 0]) == follow_id:

                        # find the frame where the lane change happened
                        if int(search_mat[j, 1]) == frame_min3:
                            h3.append(data_set[j, 3] - data_set[i-30, 3])
                            print ('the three headway is {0}'
                                   .format(data_set[j, 3] - data_set[i-30, 3]))
                            break
                        elif int(search_mat[j, 1]) == frame_min4:
                            h4.append(data_set[j, 3] - data_set[i-40, 3])
                            print ('the four headway is {0}'
                                   .format(data_set[j, 3] - data_set[i-40, 3]))
                        elif int(search_mat[j, 1]) == frame_min5:
                            h5.append(data_set[j, 3] - data_set[i-50, 3])
                            print ('the five headway is {0}'
                                   .format(data_set[j, 3] - data_set[i-50, 3]))
            print ''
        # update the id's used for testing the condition
        veh_id = int(data_set[i, 0])
        lane_id = int(data_set[i, 6])

    print h3
    print h4
    print h5
    plt.subplot(3, 1, 1)
    plt.hist(h3, bins='auto')
    plt.title("headway at 3, 4, 5 seconds, 5:15-5:30pm")
    plt.xlabel("headway")
    plt.ylabel("counts")
    plt.subplot(3, 1, 2)
    plt.hist(h4, bins='auto')
    plt.xlabel("headway")
    plt.ylabel("counts")
    plt.subplot(3, 1, 3)
    plt.hist(h5, bins='auto')
    plt.xlabel("headway")
    plt.ylabel("counts")
    plt.show()

'''We go to the point at which the lane change
is identified in the data, go back 3, 4, 5 seconds
and compute the headway of the new following car
to its predecessor at those times.
We then plot all three sets of headways.
Trajectory Data
PARAMETERS: dataset - the processed dataset from which we extract data
OUTPUT: Three histograms of headways at time of initiated lane change
'''


def lane_change_in_headway(data_set):
    headway_3 = []
    headway_4 = []
    headway_5 = []

    'initialize the system'
    veh_id = int(data_set[0, 0])
    lane_id = int(data_set[0, 6])

    'matrix of vehicle id and frame'
    'Here we are computing the probability of a car'
    'lane changing out based on the headway behind it'

    for i in range(1, data_set.shape[0]):
        # if we are still on the same vehicle and it has switched lanes
        # check that we haven't aborted the lane change within five secs
        # check makes sure the lane change has lasted at least five s
        if (int(data_set[i, 0]) == veh_id and
                lane_id != int(data_set[i, 6]) and
                lane_id != int(data_set[i+50, 6]) and
                veh_id == int(data_set[i+50, 0]) and
                int(data_set[i, 6]) != int(data_set[i-50, 6]) and
                veh_id == int(data_set[i-50, 0])):

            # if there is an observed following vehicle
            if int(data_set[i, 8]) != 0 and int(data_set[i, 7]) != 0:

                print 'vehicle is {0}, lane was {1}, ' \
                    'lane is {2}'.format(veh_id, lane_id, int(data_set[i, 6]))
                print ('the new following vehicle is {0}'
                       .format(int(data_set[i, 8])))
                print 'the leading vehicle is {0}'.format(int(data_set[i, 7]))
                # find the time at which the car is actually in the new lane
                lane_change_frame = int(data_set[i, 1])
                print 'lane change frame is {0}'.format(lane_change_frame)

                # id of the car behind it and in front after the lane change
                # Technically if there have been two lane changes
                # into this lane this will yield an error
                follow_id = int(data_set[i, 8])
                lead_id = int(data_set[i, 7])

                # the frame at which the leading car when
                # it "initiated" the lane change
                frame_min3 = int(lane_change_frame) - 30
                frame_min4 = int(lane_change_frame) - 40
                frame_min5 = int(lane_change_frame) - 50

                # now we need to find the preceeding and following vehicle
                # positions at these frames
                for j in range(1, data_set.shape[0]):

                    # we've found the following car
                    if (int(data_set[j, 0]) == follow_id and
                            int(data_set[j, 7] == lead_id)):

                        # find the frame where the lane change happened
                        if int(data_set[j, 1]) == frame_min3:
                            headway_3.append(data_set[j, 9])
                        elif int(data_set[j, 1]) == frame_min4:
                            headway_4.append(data_set[j, 9])
                        elif int(data_set[j, 1]) == frame_min5:
                            headway_5.append(data_set[j, 9])

            print ''
        # update the id's used for testing the condition
        veh_id = int(data_set[i, 0])
        lane_id = int(data_set[i, 6])

    plt.subplot(3, 1, 1)
    plt.hist(headway_3, bins='auto')
    plt.title("headway at 3, 4, 5 seconds, 4:00-4:15pm")
    plt.xlabel("headway")
    plt.ylabel("counts")
    plt.subplot(3, 1, 2)
    plt.hist(headway_4, bins='auto')
    plt.xlabel("headway")
    plt.ylabel("counts")
    plt.subplot(3, 1, 3)
    plt.hist(headway_5, bins='auto')
    plt.xlabel("headway")
    plt.ylabel("counts")
    plt.show()

if __name__ == '__main__':
    # pm_4, pm_5, pm_515 = read_data()
    # extract_cars(pm_4, '/4-415-Processed.txt')
    # extract_cars(pm_5, '/5-515-Processed.txt')
    # extract_cars(pm_515, '/515-530-Processed.txt')
    pm_4, pm_5, pm_515 = read_processed_data()
    # print num_lane_changes(pm_4)
    # informal_headway_extraction(pm_515)
    lane_change_in_headway(pm_5)
