cd '/Users/eugenevinitsky/Box Sync/Research/Bayen/Data/NGSIM-Processed'
temp_string = 'pm_230_';
data = dlmread(strcat(temp_string,'velocity_filter.txt'), ',', 1, 0);
data = horzcat(data, zeros(size(data,1), 1));
for i=1:size(data,1)
    if(mod(i, 100000) == 0)
        i 
    end
    frame = data(i,2);
    precede_id = data(i,8);
    current_y = data(i,14);
    flag = 0; 
    for j = 1:size(data,1)
        if (data(j,2) == frame && data(j,1) == precede_id ...
                && precede_id ~= 0 && current_y > 0 && data(j,14) > 0)
            precede_y = data(j,14);
            flag = 1;
            break
        end
    end
    % there is a valid proceeding car so compute the headway
    if flag == 1
        space_headway = precede_y - current_y;
        data(i,15) = space_headway;
    end
end

dlmwrite(strcat(temp_string,'velocity_diff.txt'), data, 'delimiter', ' ', ...
         'precision', 13)