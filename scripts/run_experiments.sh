




# ray exec scripts/ray_autoscale.yaml \
# "python flow/examples/train.py multiagent_i210 i210_ppo --num_iterations 40 --num_rollouts 20 \
# --checkpoint_freq 5 --use_s3 --grid_search --num_cpus 20 --algorithm PPO --rl_trainer rllib"
# --start --stop --cluster-name nathan-i210-ppo --tmux


# debug
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 i210_ppo --num_iterations 1 --num_rollouts 1 \
--checkpoint_freq 5 --use_s3 --num_cpus 20 --algorithm PPO --rl_trainer rllib" \
--start --cluster-name nathan-i210-test