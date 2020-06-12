#%% initialize the data set:
import Process_Flow_Outputs as PFO #Set of functions used to process data

csv_path  = 'PATH_TO_EMISSIONS_FILE'

csv_path = '/Users/sshanto/Vanderbilt/vis_flow/flow-visualization_tool/tutorials/data/3_lane_ring_trial_20200605-1245031591379103.272234-emission.csv'

i210_Data = PFO.SimulationData(csv_path = csv_path) #This loads the data from emissions file

expName = csv_path.split("_")[0] #Allows dynamic naming of files

#%% Specify the edges and lanes to plot

WANT_GHOST_CELL = True

#Need to specify the edges one which we want to plot data:
edge_list = ["119257914", "119257908#0", "119257908#1-AddedOnRampEdge",
                  "119257908#1", "119257908#1-AddedOffRampEdge", "119257908#2",
                  "119257908#3"]

#specify the set of lanes in which we want to plot data:
lane_list = ['4','4','5','4','5','4','5'] #Corresponds to outter most lanes

if(WANT_GHOST_CELL):
    edge_list.insert(0,'ghost0')
    lane_list.insert(0,'4')
   
    
    
    
#Specify plotting parameters:
    
time_range = [0,1600] #Specify the time range over which to plot

if(WANT_GHOST_CELL):
    pos_range = [0,2250]
else:
    pos_range = [500,2250]
    
clim=[0,30]
fileName = expName+'_SpaceTime.png'
marker_size=1.0

coloring_Attribute = 'SPEED'

#%% Run plotting code:
i210_Data.plot_Time_Space(coloring_Attribute=coloring_Attribute,edge_list=edge_list,lane_list=lane_list,clim=clim,fileName=fileName,time_range=time_range,pos_range=pos_range,marker_size=marker_size)
    
