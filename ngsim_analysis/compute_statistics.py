from pudb import set_trace
import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

'Change these to your desired paths to the NGSIM data!'
input_folder_80 = "/Users/eugenevinitsky/Box Sync/Research/Bayen/Data" \
           "/NGSIM-Raw/I-80-Main-Data/vehicle-trajectory-data"
# this one is for the US 101 data
input_folder_101= "/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/NGSIM" \
            "/US-101/vehicle-trajectory-data"
input_folder_80_proto = "/Users/eugenevinitsky/Box Sync/Research" \
            "/Bayen/Data/NGSIM/I-80prototype/vehicle-trajectory-data"
output_folder = "/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/" \
            "NGSIM-Processed"
disappear_suffix = "_disappear_headway.txt"
appear_suffix = "_appear_headway.txt"

''' Read in data that has been processed in the format of extract_cars for I-80
PARAMETERS: NONE
OUTPUT: three sets of data corresponding to 4-4:15, 5-5:15, 5:15-5:30 '''


def read_processed_data_80():
    global output_folder

    pm_4 = np.loadtxt(output_folder+"/4-415-Processed.txt", delimiter=',',
                      skiprows=1)
    pm_5 = np.loadtxt(output_folder+"/5-515-Processed.txt", delimiter=',',
                      skiprows=1)
    pm_515 = np.loadtxt(output_folder+"/515-530-Processed.txt", delimiter=',',
                        skiprows=1)

    return pm_4, pm_5, pm_515


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


def read_processed_data_80_proto():
    global output_folder

    pm_230 = np.loadtxt(output_folder+"/230-Processed.txt", delimiter=',',
                      skiprows=1)

    return pm_230

def open_file(file_name):
    return np.loadtxt(file_name)

def anderson_darling(headway, dist='norm'):
    stat, crit, significance = scipy.stats.anderson(headway, dist)
    print stat
    print crit
    print significance
    return (stat, crit, significance)

''' Computes a lognormal fit without using scipy as a check'''
def lognfit(headway):
    log_h = np.log(headway)
    mu = log_h.mean()
    sig = log_h.std(ddof=0)
    print mu, sig

def ks_tester(headway, dist_name='lognorm'):
    log_h = np.log(headway)
    param = [0, 0, 0]
    if dist_name=='lognorm':
        param[0] = log_h.std(ddof=0)
        param[1] = 0 
        param[2] = np.exp(log_h.mean())
        print 's is {0}, scale is {1}'.format(param[0], param[2])
    else: 
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(headway)
    print scipy.stats.kstest(headway, dist_name, args=param)


def filter(headway, u_cutoff=400.0, l_cutoff = 0.2):
    headway = headway[np.where(headway < u_cutoff)]
    headway = headway[np.where(headway > l_cutoff)]
    return headway


if __name__ == '__main__':
    file_set = 'pm_230'
    h_string = output_folder + '/' + file_set + appear_suffix
    #pm_4, pm_5, pm_515 = read_processed_data_80()
    # pm_750, pm_805, pm_820 = read_processed_data_101()
    pm_230 = read_processed_data_80_proto()
    l_headways = open_file(h_string)
    l_headways = filter(l_headways)
    t_headways = pm_230[:, 9]
    t_headways = filter(t_headways)
    print len(l_headways)
    print len(t_headways)
    print scipy.stats.ks_2samp(l_headways, t_headways)
    #anderson_darling(np.log(l_headways))
    #anderson_darling(np.log(filter(t_headways)))
    ks_tester(l_headways, 'burr')
    ks_tester(t_headways, 'burr')


