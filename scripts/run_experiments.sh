# run cluster without training
ray exec ray_autoscale.yaml "echo OK" --start --cluster-name nathan-worker --tmux

# local testing
python examples/train.py multiagent_i210 --exp_title i210-test --num_iterations 1 --num_rollouts 1

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
--start --cluster-name nathan-i210_test_leaderboard_0p1

# 18/07/20 -- default speed reward, penalize_stops and penalize_accel
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan 10%_ppo_speedrwd_pstops_paccel --exp_title i210_ppo_0p1_speedrwd_pstops_paccel" \
--start --cluster-name nathan-i210_ppo_0p1_speedrwd_pstops_paccel --tmux