close all
cd '/Users/eugenevinitsky/Box Sync/Research/Bayen/Code'
stability_map = load('alternate_stability.out');
[x_size, y_size] = size(stability_map);
xticks = linspace(1, x_size, 10)
x_tick_label = linspace(0,2,x_size); 

figure()
imagesc(stability_map)
colormap(gray)
title('stability map for alternating manual-automated system')
xlabel('kd')
ylabel('kv')

stability_manual_map = load('manual_stability.out');
[x_size, y_size] = size(stability_manual_map);
xticks = linspace(1, x_size, 10)
x_tick_label = linspace(0,2,x_size); 

figure()
imagesc(stability_manual_map)
colormap(gray)
xlabel('kd')
ylabel('kv')
title('stability map for fully manual system')

stability_mrandom_map = load('alternate_stability_random.out');
[x_size, y_size] = size(stability_manual_map);
xticks = linspace(1, x_size, 10)
x_tick_label = linspace(0,2,x_size); 

figure()
imagesc(stability_manual_map)
colormap(gray)
xlabel('kd')
ylabel('kv')
title('stability map for random system')