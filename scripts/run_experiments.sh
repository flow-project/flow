# run cluster without training
ray exec ray_autoscale.yaml "echo OK" --start --cluster-name nathan-submitter --tmux

# local testing
python examples/train.py multiagent_i210 --exp_title i210-test --num_iterations 1 --num_rollouts 1 --render

# 15/07/20 -- train with mpg reward, penalize_stops and penalize_accel
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 i210_ppo_0p1_mpgrwd_pstops_paccel --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search \
--upload_graphs Nathan 10%_ppo_mpgrwd_pstops_paccel --multi_node" \
--start --cluster-name nathan-i210-ppo-0p1-mpgrwd-pstops-paccel --tmux

# 18/07/20 -- test automatic leaderboard upload
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 1 --num_rollouts 1 \
--checkpoint_freq 1 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan test_leaderboard_0p1 --exp_title i210_test_leaderboard_0p1" \
--start --cluster-name nathan-i210_test_leaderboard2_0p1

# 20/07/20 01h20 -- 10%, default local speed reward, penalize_stops and penalize_accel
# fixed version (the one before was 5% with no penalty)
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan 10%_ppo_speedrwd_pstops_paccel_fixed --exp_title i210_ppo_0p1_speedrwd_pstops_paccel_fixed" \
--start --cluster-name nathan-i210_ppo_0p1_speedrwd_pstops_paccel_fixed --tmux

# 20/07/20 01h26 -- 10%, default local speed reward + brut mpg reward, no penalty
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan 10%_ppo_speed_mpg_rwd_nop --exp_title i210_ppo_0p1_speed_mpg_rwd_nop" \
--start --cluster-name nathan-i210_ppo_0p1_speed_mpg_rwd_nop --tmux

#############
#############
#############

# 22/07/20 0h28 -- speed rwd only, 10%, both penalties, removed speed curriculum and set look_back_length to 8
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan 10%_ppo_speed_ae7ue --exp_title i210_ppo_0p1_speed_ae7ue" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speed_ae7ue --tmux

# 22/07/20 12h22 -- same but reroute_on_exit = True (ran with buggy reroute)
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan 10%_ppo_speed_reroute_f9kd2 --exp_title i210_ppo_0p1_speed_reroute_f9kd2" \
--start --cluster-name nathan-i210_ppo_0p1_speed_reroute_f9kd2 --tmux

# 22/07/10 18h24 -- try speedrwd + mpgrwd/100
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan 10%_ppo_speedmpg_9ik2e --exp_title i210_ppo_0p1_speedmpg_9ik2e" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speedmpg_9ik2e --tmux

# 22/07/10 18h37 -- same but reroute_on_exit = True (ran with buggy reroute)
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan 10%_ppo_speedmpg_reroute_kzo34 --exp_title i210_ppo_0p1_speedmpg_reroute_kzo34" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speedmpg_reroute_kzo34 --tmux

# 23/07/10 2h50 -- same but without buggy reroute
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan 10%_ppo_speedmpg_reroute_abe9n --exp_title i210_ppo_0p1_speedmpg_reroute_abe9n" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speedmpg_reroute_abe9n --tmux

# 23/07/10 3h00 -- try no_done_at_end fix (with real exit_edge)
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 60 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--exp_title i210_debug_no_done_at_end" \
--start --cluster-name nathan-i210_debug_no_done_at_end_8 --tmux

###

# instant_mpg changed by dividing by 0.05
# 25/07/20 03h30 -- 10% ; reroute ; mpg/10 + speed ; no penalties   (two of these actors died between 30 and 40 iterations)
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan i210_ppo_0p1_speed1mpg10_reroute_nop_kdq34 --exp_title i210_ppo_0p1_speed1mpg10_reroute_nop_kdq34" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speed1mpg10_reroute_nop_kdq34 --tmux

# 25/07/20 03h35 -- 10% ; no reroute ; mpg/10 + speed ; no penalties
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan i210_ppo_0p1_speed1mpg10_nop_a7uj8 --exp_title i210_ppo_0p1_speed1mpg10_nop_a7uj8" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speed1mpg10_nop_a7uj8 --tmux

# 25/07/20 03h42 -- 10% ; reroute ; mpg/10 + speed ; both penalties
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan i210_ppo_0p1_speed1mpg10_reroute_bothp_j9fjea --exp_title i210_ppo_0p1_speed1mpg10_reroute_bothp_j9fjea" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speed1mpg10_reroute_bothp_j9fjea --tmux

# 25/07/20 03h50 -- 10% ; no reroute ; mpg/10 + speed ; both penalties
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan i210_ppo_0p1_speed1mpg10_bothp_nsqd98 --exp_title i210_ppo_0p1_speed1mpg10_bothp_nsqd98" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speed1mpg10_bothp_nsqd98 --tmux

###

# all the same but mpg/10 -> mpg
# these are also going with no_done_at_end = True
# 26/07/20 11h54 -- 10% ; reroute ; mpg + speed ; no penalties 
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan i210_ppo_0p1_speed1mpg1_reroute_nop_l0dj7 --exp_title i210_ppo_0p1_speed1mpg1_reroute_nop_l0dj7" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speed1mpg1_reroute_nop_l0dj7 --tmux

# 26/07/20 11h59 -- 10% ; no reroute ; mpg + speed ; no penalties
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan i210_ppo_0p1_speed1mpg1_nop_pmu87 --exp_title i210_ppo_0p1_speed1mpg1_nop_pmu87" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speed1mpg1_nop_pmu87 --tmux

# 26/07/20 12h03 -- 10% ; reroute ; mpg + speed ; both penalties
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan i210_ppo_0p1_speed1mpg1_reroute_bothp_jcv2d --exp_title i210_ppo_0p1_speed1mpg1_reroute_bothp_jcv2d" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speed1mpg1_reroute_bothp_jcv2d --tmux

# 26/07/20 12h11 -- 10% ; no reroute ; mpg + speed ; both penalties
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan i210_ppo_0p1_speed1mpg1_bothp_rck11 --exp_title i210_ppo_0p1_speed1mpg1_bothp_rck11" \
--start --stop --cluster-name nathan-i210_ppo_0p1_speed1mpg1_bothp_rck11 --tmux





# metrics: policy_reward_mean avg_speed_mean avg_speed_avs_mean avg_accel_avs_mean avg_mpg_per_veh_mean num_cars_mean

# try:
# no speed curriculum
# mix mpg reward and speed reward
# have reroute_on_exit both on and off
# try higher values for look_back_length (nb of vehicles behind the avs can see)
# divide by 0.05 in instantaneous_mpg