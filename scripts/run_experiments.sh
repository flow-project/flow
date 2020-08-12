# run cluster without training
ray exec ray_autoscale.yaml "echo OK" --start --cluster-name nathan-submitter --tmux

# local testing
python examples/train.py multiagent_i210 --exp_title i210-test --num_iterations 1 --num_rollouts 1 --render

# 26/07/20 13h -- 10% ; reroute ; mpg/100 ; stop p
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --num_iterations 40 --num_rollouts 20 \
--checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
--upload_graphs Nathan i210_ppo_0p1_mpg100_reroute_stopp_ak00p --exp_title i210_ppo_0p1_mpg100_reroute_stopp_ak00p" \
--start --stop --cluster-name nathan-i210_ppo_0p1_mpg100_reroute_stopp_ak00p --tmux

