close all
cd '/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/NGSIM-Processed'
temp_string = 'pm_750_';
if strcmp(temp_string, 'pm_230_')
    u_v = 130;
    u_h = 200;
else
    u_h = 60;
    u_v = 30;   
end
if strcmp(temp_string, 'pm_230_')
    cut_off = 5; 
else
    cut_off = 2; 
end
l_h = 3;
l_v = 10; 
data = dlmread(strcat(temp_string,'disappear_headway_velocity.txt'), ',', 1, 0);
headway = .3048*data(:, 1); 
velocity = data(:, 2);

temp_mask = bitand(headway > l_h, velocity > l_v); 
temp_mask2 = bitand(headway < u_h, velocity < u_v); 
final_mask = bitand(temp_mask, temp_mask2);
headway = headway(final_mask);
velocity = velocity(final_mask);

figure()
X = [log(headway), velocity];
hist3(X);

figure()
histogram(headway)
title('total headway')
xlabel('headway (m)')
ylabel('counts')

figure()
histogram(log(headway))
title('total log headway')

figure()
[f,xi] = ksdensity(log(headway));
plot(xi,f)
title('kde log(headway) total')

figure()
histogram(X(:,2))
title('total velocity')

[f,xi] = ksdensity(X(:,2));
plot(xi,f)
title('kde velocity total')

figure()
gkde2(X)
title('kde total')

mean(X)
cov(X)
skewness(X)
kurtosis(X)

Xtemp = X(1:floor(size(X,1)/cut_off),:);
disp('cut off values')
mu = mean(Xtemp)
sig = cov(Xtemp)
skewness(Xtemp)
kurtosis(Xtemp)

figure()
histogram(Xtemp(:,1))
title('first couple minutes headway')

[f,xi] = ksdensity(Xtemp(:,1));
plot(xi,f)
title('kde headway 7.5 min')
xlabel('headway (m)')
ylabel('counts')

figure()
histogram(Xtemp(:,2))
title('first couple minutes velocity')

[f,xi] = ksdensity(Xtemp(:,2));
plot(xi,f)
title('kde velocity 7.5 min')
xlabel('velocity (m/s)')
ylabel('counts')

figure()
gkde2(Xtemp);



figure()
[bandwidth,density,Xt,Yt]=kde2d(Xtemp);
  % plot the data and the density estimate
  contour3(Xt,Yt,density,50), hold on
  plot(X(:,1),X(:,2),'r.','MarkerSize',5)
  
figure()
h_grid = min(Xtemp(:,1)):.1:max(Xtemp(:,1)); 
v_grid = min(Xtemp(:,2)):.5:max(Xtemp(:,2)); 
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
ksdensity(Xtemp, xi)
colormap default
alpha(.5)


figure()
h_grid = min(X(:,1)):.1:max(X(:,1)); 
v_grid = min(X(:,2)):.5:max(X(:,2)); 
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
ksdensity(X, xi)
colormap default
alpha(.5)

figure()
[x1,x2] = meshgrid(h_grid, v_grid);
x1 = x1(:);
x2 = x2(:);
xi = [x1 x2];
ksdensity(X, xi)
colormap default
alpha(.5)

figure()
[x1,x2] = meshgrid(h_grid, v_grid);
x1 = x1(:);
x2 = x2(:);
xi = [x1 x2];
ksdensity(Xtemp, xi)
colormap default
alpha(.5)