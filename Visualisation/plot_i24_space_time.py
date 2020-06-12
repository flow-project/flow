#%% initialize the data set:
import Process_Flow_Outputs as PFO #Set of functions used to process data
from flow.networks.i24_subnetwork import EDGES_DISTRIBUTION

#change path name
csv_path = '/Users/sshanto/Vanderbilt/vis_flow/flow-visualization_tool/examples/data/I-24_subnetwork_20200608-1946201591663580.734905-emission.csv'

i210_Data = PFO.SimulationData(csv_path = csv_path) #This loads the data from emissions file

expName = "I-24"#Allows dynamic naming of files

#%% Specify the edges and lanes to plot

#Need to specify the edges one which we want to plot data:
edge_list = EDGES_DISTRIBUTION 

#specify the set of lanes in which we want to plot data:
lane_list = ['0','1','2','3'] #Corresponds to outter most lanes

#Specify plotting parameters:
time_range = [0,100] #Specify the time range over which to plot
pos_range = [0,1200]

clim=[0,30]
fileName = expName+'_SpaceTime.png'
marker_size=1.0

coloring_Attribute = 'SPEED'

#%% Run plotting code:
i210_Data.plot_Time_Space(coloring_Attribute=coloring_Attribute,edge_list=edge_list,lane_list=lane_list,clim=clim,fileName=fileName,time_range=time_range,pos_range=pos_range,marker_size=marker_size)
    
