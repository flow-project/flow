%%
% Process Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
cd '/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/NGSIM-Processed'
temp_string = 'pm_820_';
data_l = dlmread(strcat(temp_string,'disappear_headway_velocity.txt'), ',', 1, 0);
headway = .3048*data_l(:, 1); 
headway_mask = headway > 0;
velocity = data_l(:, 2);
headway = headway(headway_mask);
velocity = velocity(headway_mask);

% Now open the total set of lane changes
close all
cd '/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/NGSIM-Processed'
data_tot = dlmread(strcat(temp_string,'velocity_filter.txt'), ',', 1, 0);
headway_tot = .3048*data_tot(:, 1); 
headway_mask = headway_tot > 0;
headway_tot = headway_tot(headway_mask);
velocity_tot = data_tot(:, end);
velocity_tot = velocity_tot(headway_mask);
%% Edit 

% First lets compute lane changes per mile per minute
if strcmp('pm_230', temp_string) 
    lane_minute_mile = size(headway,1)/30*(5280/2950);
elseif strcmp('pm_4', temp_string) || strcmp('pm_5', temp_string) ||...
        strcmp('pm_515', temp_string)
    lane_minute_mile = size(headway,1)/15*(5280/1650);
else
    lane_minute_mile = size(headway,1)/15*(5280/2100);
end

disp(sprintf('lane change per minute per mile is %f', lane_minute_mile))

% Now lets compute the ratio of cars that lane change to the total
lane_change_ratio = size(unique(data_l(:,3)),1)/size(unique(data_tot(:,1)),1);
disp(sprintf('ratio of cars that lane change to the total is %f',...
    lane_change_ratio))

% check that the cars only lane change once rule is approximately true
size(data_l(:,3))