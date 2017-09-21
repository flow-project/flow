import numpy as np

'Change this to your path to the the NGSIM data!'
prefix = "/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/NGSIM/I-80/vehicle-trajectory-data"

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
    global prefix

    pm_4 = np.loadtxt(prefix+"/0400pm-0415pm/RECONSTRUCTED-trajectories-400-0415_NO_MOTORCYCLES.csv",
        delimiter=',', skiprows=1)
    pm_5 = np.loadtxt(prefix+"/0500pm-0515pm/trajectories-0500-0515.csv",
        delimiter=',', skiprows=1)
    pm_515 = np.loadtxt(prefix+"/0515pm-0530pm/trajectories-0515-0530.csv",
        delimiter=',', skiprows=1)

    return pm_4,pm_5,pm_515

''' Outputs the number of lane changes that occur in a data set.
    The function looks at the vehicle id and checks if for a given vehicle ID
    whether there are two lane numbers. 
    This function was just to confirm that our data set does contain lane changes. 

    PARAMETERS: data_set- the array we are checking for lane changes
    OUTPUT: The number of lane changes that occured in this data set
'''
def num_lane_change(data_set):
    num_lane_changes = 0

    'initialize the system'
    veh_id = data_set[0,0]
    lane_id = data_set[0,2]
    for i in range(1,data_set.shape[0]):
        'if we are still on the same vehicle and the lane has changed'
        if data_set[i,0] == veh_id and lane_id != data_set[i, 2]:
            num_lane_changes = num_lane_changes + 1
            print 'vehicle is {0}, lane was {1}, lane is {2} \n'.format(veh_id, lane_id, data_set[i, 2])
        veh_id = data_set[i,0]
        lane_id = data_set[i,2]

    return num_lane_changes

if __name__=='__main__':
    pm_4, pm_5, pm_515 = read_data()
    print num_lane_change(pm_4)
