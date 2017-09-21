close all
cd '/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/NGSIM-Processed'
temp_string = 'pm_750_';
if strcmp(temp_string, 'pm_230_')
    u_v = 130;
    u_h = 200;
else
    u_h = 120;
    u_v = 60;   
end
if strcmp(temp_string, 'pm_230_')
    cut_off = 5; 
else
    cut_off = 3; 
end
l_h = 3;
l_v = 1; 
data = dlmread(strcat(temp_string,'appear_headway_velocity.txt'), ',', 1, 0);
headway = .3048*data(:, 1); 
velocity = data(:, 2);

temp_mask = bitand(headway > l_h, velocity > l_v); 
temp_mask2 = bitand(headway < u_h, velocity < u_v); 
final_mask = bitand(temp_mask, temp_mask2);
headway = headway(final_mask);
velocity = velocity(final_mask);

time_h = headway./velocity;
u_h = 12;
l_h = 0;
h_mask = bitand(time_h > l_h, time_h < u_h);
time_h = time_h(h_mask);

figure()
[f,xi] = ksdensity(time_h);
plot(xi,f)

figure()
histfit(time_h, [], 'burr')

figure()
histogram(time_h)
