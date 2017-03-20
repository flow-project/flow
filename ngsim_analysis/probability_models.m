function probability_models
close all
global L beta  delta_t
L = 230;
%beta = 4.5618;
beta = .9825;
delta_t = 4; 
h = .01:.1:100;
[pa,pd] = probability_headway(h);
figure()
plot(h, pa)
hold on
plot(h, pd)
legend('appear', 'disappear')
trapz(h, pa)

n = 3:1:100;
h = L./n;
[pa,pd] = probability_headway(h);
figure()
plot(n, pa)
hold on
plot(n, pd)
legend('appear', 'disappear')

trapz(n, pa)

n = 4:1:50;
trans = compute_transition_mat(min(n), max(n))
% normalize the transition_matrix
for i = 1:max(size(trans))
    trans(i,:) = trans(i,:)/sum(trans(i,:));
end
% find the steady states
plot_trans(trans)
% xi = rand(size(trans,1),1)
% xi = xi/sum(xi)
%xi = zeros(size(trans,1),1)
%xi(13) = 1; 
xi = ones(size(trans,1),1)/size(trans,1);
trans_temp = trans;
for i = 1:800000
    xi = trans*xi;
    trans_temp = trans_temp*trans;
end
xi;
trans_temp(3,:)

% average number of cars
trans_temp(3,:)*n'
% average headway
trans_temp(3,:)*(L./n)'
v = 15*(1 - cos(pi*(L./n - 5)/30));
% average velocity
trans_temp(3,:)*(v')

[pa, pd] = probability_headway(L./n);
% expected number of lane changes per 230 km UNITS ARGH
% use this to calibrate time-step
trans_temp(3,:)*(n.*pa)'
trans_temp(3,:)*(n.*pd)'
end

function plot_trans(trans_map)
figure()
imagesc(trans_map)
colorbar()
end

function [pa, pd] = probability_headway(h)
    % these data points correspond to the 70 dataset
    global beta delta_t
    %mu_appear = 3;
    mu_appear = 3.09;
    mu_disappear = 2.79;
    mu_tot = 2.95;
    sig_appear = .3642;
    sig_disappear = .4176;
    sig_tot = .4255;
    
    c_appear = normalization_constant(mu_appear, sig_appear, mu_tot, sig_tot);
    c_disappear = normalization_constant(mu_disappear, sig_disappear, mu_tot, sig_tot);
    p_appear = exp(((log(h) - mu_tot).^2)/(2*sig_tot^2) - ((log(h) - mu_appear).^2)/(2*sig_appear^2));
    p_disappear = exp(((log(h) - mu_tot).^2)/(2*sig_tot^2) - ((log(h) - mu_disappear).^2)/(2*sig_disappear^2));
     pa = c_appear*p_appear*(delta_t/beta);
     pd = c_disappear*p_disappear*(delta_t/beta); 

end

% the 1 index corresponds to the appear/disappear
function c = normalization_constant(mu1, sig1, mu2, sig2)
    coeff = (sqrt((1/sig1^2) - (1/sig2^2)))/(sqrt(2*pi));
    exp_factor = exp(((mu1 - mu2)^2 - 2*mu2*sig1^2 + 2*mu1*sig2^2 + (sig1*sig2)^2)/(2*sig1^2 - 2*sig2^2));
    c = coeff*exp_factor;
end

function tn_np = transition_mat_component(n, np)
    global L
    h = L/n;
    [pa, pd] = probability_headway(h);
    if np > 2*n
        tn_np = 0;
    elseif np >= n
        delta = np - n;
        tn_np = 0;
        for i = 0: n - delta
            tn_np = tn_np + nchoosek(n, i+delta)*(pa^(i+delta))*((1-pa)^(n-delta-i))*...
                nchoosek(n, i)*(pd^i)*((1-pd)^(n-i));
        end    
    else
        delta = abs(np - n);
        tn_np = 0;
        for i = 0: n - delta
            tn_np = tn_np + nchoosek(n, i)*(pa^(i))*((1-pa)^(n-i))*...
                nchoosek(n, i+delta)*(pd^(i+delta))*((1-pd)^(n-delta-i));
        end    
    end
end

function trans_mat = compute_transition_mat(n_min, n_max)
    trans_mat = zeros(n_max - n_min + 1, n_max - n_min + 1);
    for i = n_min:n_max
        for j = n_min:n_max
            i
            trans_mat(i-n_min+1, j-n_min+1) = transition_mat_component(i, j);
        end
    end
end