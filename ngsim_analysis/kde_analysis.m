function kde_analysis
close all
cd '/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/NGSIM-Processed'
temp_string = 'pm_750_';
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
data = dlmread(strcat(temp_string,'velocity_filter.txt'), ',', 1, 0);
headway = .3048*data(:, 10); 
velocity = data(:, 14);
temp_mask_l = bitand(headway > l_h, velocity > l_v);
temp_mask_u = bitand(headway < u_h, velocity < u_v);  
temp_mask = bitand(temp_mask_l, temp_mask_u);
headway = headway(temp_mask);
velocity = velocity(temp_mask);

X = [log(headway), velocity];

figure()
histogram(headway)
title('total headway')
xlabel('headway (m)')
ylabel('counts')

figure()
histogram(log(headway))
title('total log headway')
xlabel('log(headway) (m)')
ylabel('counts')

figure()
[f,xi] = ksdensity(log(headway));
plot(xi,f)
title('kde log(headway) total')
xlabel('log(headway) (m)')
ylabel('counts')

figure()
histogram(X(:,2))
title('total velocity')
xlabel('velocity (m/s)')
ylabel('counts')

[f,xi] = ksdensity(X(:,2));
plot(xi,f)
title('kde velocity total')
xlabel('velocity (m/s)')
ylabel('counts')

figure()
[f,xi] = ksdensity(velocity(1:floor(size(velocity,1)/cutoff)));
plot(xi,f)
title(sprintf('first %d minutes, velocity (m/s)', 15/cutoff))
xlabel('velocity (m/s)')
ylabel('counts')

figure()
[f,xi] = ksdensity(headway(1:floor(size(headway,1)/cutoff)));
plot(xi,f)
title(sprintf('first %d minutes, headway (m)', 15/cutoff))
xlabel('headway (m)')
ylabel('counts')

figure()
[f,xi] = ksdensity(log(headway(1:floor(size(headway,1)/cutoff))));
plot(xi,f)
title(sprintf('first %d minutes, headway (m)', 15/cutoff))
xlabel('log(headway (m))')
ylabel('counts')

figure()
histogram(log(headway(1:floor(size(headway,1)/cutoff))));
title(sprintf('first %d minutes, headway (m)', 45/cutoff))
xlabel('log(headway (m))')
ylabel('counts')

mean(X)
cov(X)
skewness(X)
kurtosis(X)

figure()
% pull out only the front half, where conditions are probably pretty
% regular
size(X,1);
hist3(X);

ordering = randperm(size(X,1));
Xrand = X(ordering, :);

figure()
gkde2(Xrand(1:sample_num, :));

% Now we split the data into third
X = X(1:floor(size(X,1)/cutoff),:);

ordering = randperm(size(X,1));
Xrand = X(ordering, :);

figure()
histogram(Xrand(:,2))

% finally compute statistics 
Xt = Xrand(1:sample_num, :);
mu = mean(Xt)
sig = cov(Xt)
skewness(Xt)
kurtosis(Xt)

figure()
gkde2(Xrand(1:sample_num, :));
 

% create a grid
figure()
h_grid = min(Xt(:,1)):.1:max(Xt(:,1)); 
v_grid = min(Xt(:,2)):.5:max(Xt(:,2)); 
[X1,X2] = meshgrid(h_grid, v_grid);
F = mvnpdf([X1(:) X2(:)],mu,sig);
F = reshape(F,length(v_grid),length(h_grid));
surf(h_grid,v_grid,F);
alpha(.3)
colormap winter
hold on

 
[x1,x2] = meshgrid(h_grid, v_grid);
x1 = x1(:);
x2 = x2(:);
xi = [x1 x2];
ksdensity(Xt, xi)
colormap default
%colormap([1 0 0;0 0 1])
alpha(.7)

end
