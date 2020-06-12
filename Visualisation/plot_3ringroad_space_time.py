"""
idea:
    cmd inputs: filename, nameOffigure
"""

import Process_Flow_Outputs as PFO #Set of functions used to process data

csv_path = '/Users/sshanto/Vanderbilt/vis_flow/flow-visualization_tool/tutorials/data/3_lane_ring_trial_20200605-1356161591383376.051417-emission.csv'

threeRingRoad_Data = PFO.SimulationData(csv_path = csv_path) #This loads the data from emissions file

expName = csv_path.split("_")[0] #Allows dynamic naming of files

#%% Specify the edges and lanes to plot

WANT_GHOST_CELL = False

#Need to specify the edges one which we want to plot data:
edge_list = ["bottom","right","top","left"]

#specify the set of lanes in which we want to plot data:
lane_list = ['0','1','2','4'] 

#Specify plotting parameters:

time_range = [0,300] #Specify the time range over which to plot

pos_range = [0,8000]

clim=[0,30]
fileName = expName+'_SpaceTime.png'
marker_size=1.0

coloring_Attribute = 'SPEED'

#%% Run plotting code:
threeRingRoad_Data.plot_Time_Space(coloring_Attribute=coloring_Attribute,edge_list=edge_list,lane_list=lane_list,clim=clim,fileName=fileName,time_range=time_range,pos_range=pos_range,marker_size=marker_size)
