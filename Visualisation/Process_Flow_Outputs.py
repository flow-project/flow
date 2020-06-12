import numpy as np
import matplotlib.pyplot as pt
import csv
from copy import deepcopy


class SimulationData():
    
    def __init__(self,csv_path):
        self.csv_path = csv_path
        self.veh_ids = []
        self.data_ids = []
        self.data_integrals = {}
        self.SimulationData_Dict = {}
        self.Extract_Data()
        print('Data successfully loaded.')
            
        
        
    def Extract_Data(self):
        '''
        Extracts all of the data from the emissions file into a dictionary and then
        for each vehicle organizes each data field into its own list inside of an
        inner dictionary. Can reference the data for a given field via:
        SimulationData_Dict[veh_id][data_id]  
        '''
        SimulationData_Dict = {}
        line_counter = 0
        time_series_labels = []
        time_series_indices = []
        
        with open(self.csv_path) as emissions_data_csv:
            emissions_data_reader = csv.reader(emissions_data_csv,delimiter=',')
            for row in emissions_data_reader:
                #Iterate throug each row of the csv
                if(line_counter == 0):
                    time_series_labels = deepcopy(row)

                    #NOT CORRECT
                    time_series_indices = list(np.linspace(0,len(time_series_labels)-1,len(time_series_labels)).astype(int))                    
                    
                    #take out the veh ids which are used as keys in the dictionaries
                    del time_series_labels[0]
                    del time_series_labels[5] 
                    del time_series_labels[6]
                    
                    del time_series_indices[0]
                    del time_series_indices[5] 
                    del time_series_indices[6]
                    
                    
                    for i in range(len(time_series_labels)):
                        time_series_labels[i] = time_series_labels[i].upper()
                        

                    #makes the veh_id key out of the type and the id
                    
                    line_counter += 1
                    
                else:
                    # Make the veh_id:
                    veh_id = row[5]+'-'+row[6]
                    
                    if(veh_id in SimulationData_Dict.keys()):
                        # add each time 
                        time = row[0]
                        
                        type_counter = 0
                        for type_id in time_series_labels:
                            #Add each
                            type_id_index = time_series_indices[type_counter]
                            data = row[type_id_index]
                            SimulationData_Dict[veh_id][type_id][0].append(time)
                            SimulationData_Dict[veh_id][type_id][1].append(data)
                            
                            type_counter += 1
           
                    else:
                        #First data point for a certain vehicle:
                        SimulationData_Dict[veh_id] = {}
                        time = row[0]
                        
                        type_counter = 0
                        for type_id in time_series_labels:
                            type_id_index = time_series_indices[type_counter]
                            data = row[type_id_index]
                            data = row[type_id_index]
                            SimulationData_Dict[veh_id][type_id] = [[time],[data]]
                            
                            type_counter += 1
                            
        self.veh_ids = list(SimulationData_Dict.keys())
        self.data_ids = time_series_labels

        self.SimulationData_Dict = SimulationData_Dict
        
                
        self.data_ids.append('TOTAL_POSITION')
        
        total_position_dict = self.get_Total_Pos()
        
        for veh_id in self.veh_ids:     
            self.SimulationData_Dict[veh_id]['TOTAL_POSITION']=total_position_dict[veh_id]
            
#        for n in range(len(self.data_ids)):
#            self.data_ids[n] = self.data_ids[n].upper()
    
    def get_Timeseries_Dict(self,data_id=None,want_Numpy=False):
        '''
        Returns a dictionary of all timeseries for a given data field, specified by
        the data_id input. want_Numpy=True will return each timeseries organized in a 
        numpy array.
        '''
        
        data_id = data_id.upper()
        
        timeseries_dict = dict.fromkeys(self.veh_ids)
        
        if(data_id not in self.data_ids):
            raise Exception('Specificied data type not available')
        
        for v in self.veh_ids:
            if(want_Numpy):
                timeseries_dict[v] = np.array(self.SimulationData_Dict[v][data_id]).astype(float)
            else:
                timeseries_dict[v] = self.SimulationData_Dict[v][data_id]
            
        return timeseries_dict
    
    
    def get_Total_Pos(self):
        '''
        This function uses the relative position measurements provided in the emissions
        file, which are only for each edge the vehicle passes along, to calculat the
        position of the vehicle along its entire journey.
        '''
        
        total_position_dict = dict.fromkeys(self.veh_ids)
        
        edge_dict = self.get_Timeseries_Dict(data_id = 'edge_id')
        
        rel_position_dict = self.get_Timeseries_Dict(data_id = 'relative_position',want_Numpy=True)
        
        for veh_id in self.veh_ids:
            #iterate through all vehicles
                 
            rel_pos_data =  rel_position_dict[veh_id]
            
            total_pos_data = deepcopy(rel_pos_data)
            
            edge_data = edge_dict[veh_id]
            
            num_steps = len(total_pos_data[0])
            
            curr_edge = edge_data[1][0]
            
            for n in range(num_steps-1):
                # add the original rel_pos to remaing when switching edges
                next_edge = edge_data[1][n+1]
                if(curr_edge != next_edge):
                    #If just switched to new edge, add the old position to all
                    #all coming positions to create the cumulative pos
                    total_pos_data[1,n+1:] += rel_pos_data[1,n]
                    
                curr_edge = next_edge
                
            total_position_dict[veh_id] =  total_pos_data
            
        return total_position_dict
        
    def plot_Time_Space(self,coloring_Attribute='speed',edge_list=None,lane_list=None,clim=None,fileName=None,time_range=[0,1000],pos_range=[0000,2000],marker_size=1.0)  :
        '''
        Plots the space-time diagram for a specified range (both over time and space) and can be colored according to
        to a specified numerical data field. By default it colors according to speed. 
        
        edge_list and lane_list should both be a list of strings that specify the edges and lanes.
        
        Additionally, one must specify the edges and lanes over which the space-time is plotted.
        '''
        
        edge_dict = self.get_Timeseries_Dict(data_id = 'EDGE_ID')

        lane_dict = self.get_Timeseries_Dict(data_id = 'LANE_NUMBER')
        
        total_position_dict = self.get_Timeseries_Dict(data_id = 'TOTAL_POSITION',want_Numpy=True)
        
        color_dict = self.get_Timeseries_Dict(data_id = coloring_Attribute,want_Numpy = True)
        
        
        time_space_list = []
        color_list = []

        for veh_id in self.veh_ids:
            edge_data = edge_dict[veh_id][1]
            lane_data = lane_dict[veh_id][1]
            color_data = color_dict[veh_id][1,:]
            
            time_data = total_position_dict[veh_id][0,:]
            pos_data = total_position_dict[veh_id][1,:]
            
            num_samples = len(time_data)
            

            
            for n in range(num_samples):
                
                lane = lane_data[n]
                edge = edge_data[n]
        
                if(edge in edge_list):
                    edge_num = edge_list.index(edge) 
                    if(lane == lane_list[edge_num]):

                        t = time_data[n]
                        x = pos_data[n]
                        c = color_data[n]
                        
                        if(t>=time_range[0] and t<=time_range[1] and x>=pos_range[0] and x<=pos_range[1]):
                        
                            time_space_list.append([t,x])
                            color_list.append(c)
                        
        data = np.array(time_space_list)
        color_data = np.array(color_list)
                        
        fig = pt.figure(figsize=(30, 30))
        
        sc = pt.scatter(data[:,0],data[:,1],s=marker_size,c=color_data,marker='.')
        pt.grid()
        pt.xlim(time_range)
        pt.ylim(pos_range)
        if(clim is not None):
            pt.clim(clim)
        pt.colorbar(sc)
        pt.xlabel('Time [s]')
        pt.ylabel('Position [m]')
        pt.show() 
        if(fileName is not None):
            fig.savefig(fileName)
            
            
    def trim_Timesries(self,data_id='SPEED',pos_range=None,time_range=None):
        '''
        This function will trim all data for vehicles in the simulation to only contain
        values that occur either within some time range, some position range, or both. It
        directly alters the stored values in the overall class. This can be undone by rerunning
        Extract_Data().
        
        NEEDS TESTING
        '''

        pos_dict = self.get_Timeseries_Dict(data_id='TOTAL_POSITION',want_Numpy=True)

        for veh_id in self.veh_ids:
            #iterate through all vehicles
            
            pos_data = pos_dict[veh_id]
            pos_range_new = pos_range
            time_range_new = time_range
            
            num_steps = len(pos_data)
            
            #If a range is not specified then it becomes the entire range
            if(pos_range is None):
                pos_range_new = [pos_data[1,0],pos_data[1,-1]]
            if(time_range is None):
                time_range_new = [pos_data[0,0],pos_data[0,-1]]
                
            for data_id in self.data_ids:
                #iterate through different data_ids
                
                new_data = []
                old_data = self.SimulationData_Dict[veh_id][data_id]
                
                for n in range(num_steps):
                    t = pos_data[0,n]
                    x = pos_data[1,n]
                    if(t>=time_range[0] and t<=time_range[1] and x>=pos_range[0] and x<=pos_range[1]):
                        #If in proper range then put in new list to be written
                        new_data.append(old_data[n])
                        
                
                self.SimulationData_Dict[veh_id][data_id] = new_data
                
        print('Data trimmed.')
                
                
                
                
                
                
            
            
            
        
            
            
    def write_Data_To_CSV(self,data_id_list=['SPEED','SPACING','TOTAL_POSITION'],filePath=None):
        '''
        Writes data from individual vehicles in to csvs. the filePath should specify the path
        to a folder where all the csvs are written to. the data_id_list should hold the data_ids
        in order that are to be included in each csv. Time is written as the first column for each
        proceeded by the data from each item in data_id_list.
        '''

        for data_id in data_id_list:
            data_id = data_id.upper()
            
        timeseries_dict_list = []
        
        for data_id in data_id_list:
            timeseries_dict = self.get_Timeseries_Dict(data_id=data_id,want_Numpy = True)
            timeseries_dict_list.append(timeseries_dict)
            
        for veh_id in self.veh_ids:
            data_list = []
            
            #Add time to the beginning:
            data_list.append(timeseries_dict_list[0][veh_id][0,:])
            
            #Add each desired data_id to be written:
            for i in range(len(timeseries_dict_list)):
                data_list.append(timeseries_dict_list[i][veh_id][1,:])
                
            data_list_numpy = np.array(data_list).T
            
            fileName = filePath+veh_id+'.csv'
            
            np.savetxt(fileName,data_list_numpy,delimiter=',')
        
        
        print('Finished writing files')
                
        
            
#    def plot_data_vs_data(self,data_id1='SPEED',data_id2='SPACING',color_data_id=None):
#        
#        '''NOT FINISHED'''
#        
#        data_id1 = data_id1.upper()
#        data_id2 = data_id2.upper()
#        color_data_id = color_data_id.upper()
#        
#        if(data_id1 not in self.data_ids or data_id2 not in self.data_ids):
#            raise Exception('Must have specificed data_ids loaded')
#        
#        timeseries_dict1 = self.get_Timeseries_Dict(data_id = data_id1,want_Numpy=True)
#        
#        timeseries_dict2 = self.get_Timeseries_Dict(data_id = data_id2,want_Numpy=True)
#        
#        plotting_data = []
#        
#
#        for veh_id in self.veh_ids:
#            x_vals = timeseries_dict1[veh_id][1,:]
#            y_vals = timeseries_dict2[veh_id][1,:]
#            
#            num_steps = len(x_vals)
#            for i in range(num_steps):
#                x = x_vals(i)
#                y = y_vals(i)
#                plotting_data.append([x,y])
#                
#        plotting_data = np.array(plotting_data)
#                
#        if(color_data_id is not None):
#            # A data-id for coloring was provided
#            timeseries_colordict = self.get_Timeseries_Dict(data_id = color_data_id,want_Numpy=True)
#            coloring_data = []
#            
#            for veh_id in self.veh_ids:
#            
#
#            sc = pt.scatter(plotting_data[:,0],plotting_data[:,1],s=1.0,c=color_data,marker='.')
#            
#        else:
#            # A data-id for coloring was not provided
#            sc = pt.scatter(plotting_data[:,0],plotting_data[:,1],s=1.0,marker='.')
            
            
            
        
        
            
        
    def get_Spacing(self,max_spacing=300.0):
        
        '''
        Calculates the space-gap for given vehicle over the simulation and adds
        that data as a field to each vehicles data. A maximum possible speed
        difference is specified by default as 300.
        
        Note: This function is needed for this release as spacing is not reported
        by SUMO/Flow by default, but this may change in subsequent updates.
        
        BUG NOTE: This only works for single lane setups. Need to expand to be able
        
        
        
        '''
        
        
        
           
        lane_dict = self.get_Timeseries_Dict(data_id = 'lane_number',want_Numpy=True)
            
        total_position_dict = self.get_Timeseries_Dict(data_id = 'total_position',want_Numpy=True)
        
        spacing_dict = {}
        
        for veh_id in self.veh_ids:
            
            times = total_position_dict[veh_id][0,:]
            
            numSteps = len(times)
            
            curr_pos_data = total_position_dict[veh_id][1,:]
            
            curr_lane_data = lane_dict[veh_id][1,:]
            
            curr_spacing = np.ones(np.shape(curr_pos_data))*max_spacing
        
            for t in range(numSteps):
                min_spacing = max_spacing
                for other_veh in self.veh_ids:
                    if(other_veh is not veh_id):
                        
                        if(t<len(total_position_dict[other_veh][1,:])):
                        #Load in other vehicle to be compared to
                            other_pos = total_position_dict[other_veh][1,t]
                            other_lane = lane_dict[other_veh][1,t]
                            #Can be compared if in the same lane
                            if(other_lane == curr_lane_data[t]):
                                pos_diff = other_pos - curr_pos_data[t]
                                if((pos_diff > 0)&(pos_diff < min_spacing)):
                                    min_spacing = pos_diff
                                    curr_spacing[t] =  min_spacing
    
            spacing_dict[veh_id] = np.array([times,curr_spacing])
            
            
        self.data_ids.append('SPACING')
        
        for veh_id in self.veh_ids:     
            self.SimulationData_Dict[veh_id]['SPACING']=spacing_dict[veh_id]
    
        print('Spacing added.')

#    def get_Rel_Vel(self):
#        
#        '''NOT COMPLETE'''
#           
#        lane_dict = self.get_Timeseries_Dict(data_id = 'lane_number',want_Numpy=True)
#            
#        total_position_dict = self.get_Timeseries_Dict(data_id = 'total_position',want_Numpy=True)
#        
#        speed_dict = self.get_Timeseries_Dict(data_id = 'speed',want_Numpy=True)
#        
#        rel_vel_dict = {}
#        
#        for veh_id in self.veh_ids:
#            
#            times = total_position_dict[veh_id][0,:]
#            
#            numSteps = len(times)
#            
#            curr_pos_data = total_position_dict[veh_id][1,:]
#            
#            curr_lane_data = lane_dict[veh_id][1,:]
#            
#            curr_speed_data = speed_dict[veh_id][1,:]
#        
#            for t in range(numSteps):
#                for other_veh in self.veh_ids:
#                    if(other_veh is not veh_id):
#                        
#                        if(t<len(total_position_dict[other_veh][1,:])):
#                        #Load in other vehicle to be compared to
#                            other_pos = total_position_dict[other_veh][1,t]
#                            other_speed = speed_dict[other_veh][1,t]
#                            other_lane = lane_dict[other_veh][1,t]
#                            #Can be compared if in the same lane
#                            if(other_lane == curr_lane_data[t]):
#                                speed_diff = other_speed - curr_pos_data[t]
#                                
#                                if((pos_diff > 0)&(pos_diff < min_spacing)):
#                                    min_spacing = pos_diff
#                                    curr_spacing[t] =  min_spacing
#    
#            spacing_dict[veh_id] = np.array([times,curr_spacing])
#            
#            
#        self.data_ids.append('SPACING')
#        
#        for veh_id in self.veh_ids:     
#            self.SimulationData_Dict[veh_id]['SPACING']=spacing_dict[veh_id]
#    
#        print('Spacing added.')






    def get_Space_Time_Integral(self,data_id='SPEED',time_range=None,pos_range=None):
        
        '''NEED TO TEST'''
        
        '''
        This function finds the integral wrt to time of a specified numerical quantity.
        An example could be calculating 
        
        '''
        
        time_range_orig = deepcopy(time_range)
        
        pos_range_orig = deepcopy(pos_range)
        
        data_id = data_id.upper()
        
        timeseries_dict = self.get_Timeseries_Dict(data_id=data_id,want_Numpy=True)
        
        pos_dict = self.get_Timeseries_Dict(data_id='TOTAL_POSITION',want_Numpy=True)
        
        running_sum = 0.0
        
        for veh_id in self.veh_ids:
            
            pos_data = pos_dict[veh_id]
            
            if(pos_range_orig is None):
                pos_range = [pos_data[1,0], pos_data[1,-1]]
            if(time_range_orig is None):
                time_range = [pos_data[0,0], pos_data[0,-1]]  
            
            vehicle_data = timeseries_dict[veh_id][1,:]
            times = timeseries_dict[veh_id][0,:]
            num_steps = len(times)
            
            for n in range(1,num_steps):
                              
                data = vehicle_data[n]
                dt = times[n]-times[n-1]
                
                t = times[n]
                x = pos_data[n]
                
                if(t>=time_range[0] and t<=time_range[1] and x>=pos_range[0] and x<=pos_range[1]):
                    running_sum += data*dt
                    
            
        self.data_integrals[data_id] = running_sum
