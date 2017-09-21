from pudb import set_trace
import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def burr(x,c,k):
    return c*k*(np.power(x, c-1))/np.power((1+np.power(x,c)), k+1)

def density_estimator(x, bins):

if __name__=='__main__':
    sample = np.loadtxt(output_folder+'/pm_750_velocity_filter.txt', 
                        delimiter=',', skiprows=1)