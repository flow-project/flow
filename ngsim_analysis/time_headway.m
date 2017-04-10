function time_headway
close all
cd '/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/NGSIM-Processed'
temp_string = 'pm_820_';
dist_string = 'lognormal'

data_tot = dlmread(strcat(temp_string,'velocity_filter.txt'), ',', 1, 0);
headway_tot = .3048*data_tot(:, 10); 
velocity_tot = data_tot(:, 14);
    
data_lc = dlmread(strcat(temp_string,'appear_headway_velocity.txt'),...
    ',', 1, 0);
headway_lc = .3048*data_lc(:, 1); 
velocity_lc = data_lc(:, 2);

if strcmp(temp_string, 'pm_230_')
    u_v = 50;
    u_h = 150;
    cutoff = 5;
else
    u_h = 80;
    u_v = 25; 
    cutoff = 2;
end
sample_num = 50000;
l_h = 1;
l_v = 1; 

temp_mask_l = bitand(headway_tot > l_h, velocity_tot > l_v);
temp_mask_u = bitand(headway_tot < u_h, velocity_tot < u_v);  
temp_mask = bitand(temp_mask_l, temp_mask_u);
headway_tot = headway_tot(temp_mask);
velocity_tot = velocity_tot(temp_mask);

temp_mask_l = bitand(headway_lc > l_h, velocity_lc > l_v);
temp_mask_u = bitand(headway_lc < u_h, velocity_lc < u_v);  
temp_mask = bitand(temp_mask_l, temp_mask_u);
headway_lc = headway_lc(temp_mask);
velocity_lc = velocity_lc(temp_mask);

time_h_tot = headway_tot./velocity_tot;
time_h_lc = headway_lc./velocity_lc;


% headway fit
pd = fitdist(headway_lc, dist_string);
disp('ks test is')
[h, p] = kstest(headway_lc, 'Alpha', 0.05, 'CDF', pd)
[h2, p2] = kstest(headway_tot, 'Alpha', .05, 'CDF', pd)

% Make histogram of both headway sets w/ overlaid fit
x_vals = 0.01:.2:max(headway_tot);
pd = fitdist(headway_tot, dist_string);
y = pdf(pd, x_vals);
figure()
ax1 = subplot(2,1,1);
hold on
h = histogram(headway_tot, 'Normalization', 'pdf');
h.FaceColor = [.1, 0, 1.0];
plot(x_vals, y, 'LineWidth', 2)
xlabel('headway (meters)')
ylabel('normalized counts')
%axis([0, 80, 0, .06])

% lc plot
ax2 = subplot(2,1,2);
x_vals = 0.01:.2:max(headway_tot);
pd = fitdist(headway_lc, dist_string);
y = pdf(pd, x_vals);
hold on
h = histogram(headway_lc, 'Normalization', 'pdf');
h.FaceColor = [.1, 0, 1.0];
plot(x_vals, y, 'LineWidth', 2)
xlabel('headway (meters)')
ylabel('normalized counts')
%axis([0, 80, 0, .035])

% ks_density of time headway total
figure()
[f,xi] = ksdensity(time_h_tot);
plot(xi,f)

% Overlay fit onto histogram of time headway
figure()
histfit(time_h_tot, [], dist_string)

% Overlay fit onto ks_density of time headway
x_vals = 0.01:.2:max(time_h_tot);
pd = fitdist(time_h_tot, dist_string);
y = pdf(pd, x_vals);
figure()
plot(xi,f)
hold on
plot(x_vals,y)

% Overlay fit onto ks_density of time headway
x_vals = 0.000001:.001:max(time_h);
pd = fitdist(time_h, dist_string);
y = pdf(pd, x_vals);
figure()
plot(xi,f)
hold on
plot(x_vals,y)

% Overlay fit onto histogram of time headway
figure()
h1 = histogram(time_h);
h1.Normalization = 'probability';
hold on
plot(x_vals, y)

figure()
cdfplot(time_h)
hold on
plot(x_vals, cdf(pd, x_vals))

% Compute the kl divergence
kl_divergence = 0;
for i = 1:size(x_vals,2)-1
    P = density_estimator(time_h, x_vals(i));
    Q = pdf(pd, x_vals(i));
    dx = x_vals(i+1) - x_vals(i); 

    if P ~= 0
        kl_divergence = kl_divergence + dx*P*log(P/Q);
    end
end
disp('kl divergence is')
disp(kl_divergence)

% Compute the squared error
sq_err = 0;
for i = 1:size(time_h,1)
    xp = density_estimator(time_h, time_h(i));
    x = pdf(pd, time_h(i));
    sq_err = sq_err + (xp - x)^2;
end
disp('squared error is')
disp(sq_err)

disp('ks test is')
kstest(time_h, 'Alpha', 0.05, 'CDF', pd)

% disp(sprintf('mean is %f', exp(pd.mu)))
% disp(sprintf('sigma is %f', pd.sigma))
end

function prob = density_estimator(data, data_point)
[N, bin_edges] = histcounts(data);
bin_index = 0;
for i = 1:size(bin_edges,2) - 1
    if data_point > bin_edges(i) && data_point < bin_edges(i+1)
        bin_index = i; 
    end
end
% probability is number of points in bin, over total points, over bin width
prob = (N(bin_index)/(sum(N)*...
    (bin_edges(bin_index+1) - bin_edges(bin_index))));

end

