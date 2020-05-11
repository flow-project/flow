#!/bin/bash

#ray exec ray_autoscale.yaml "python flow/examples/train.py multiagent_i210 i210_reroute_test --algorithm PPO \
#--num_iterations 200 --num_cpus 12 --num_rollouts 12 --rl_trainer rllib --use_s3" --start --stop \
#--cluster-name=ev_i210_test --tmux

#ray exec ray_autoscale.yaml "python flow/examples/train.py multiagent_i210 i210_reroute_test2 --algorithm PPO \
##--num_iterations 200 --num_cpus 4 --num_rollouts 4 --rl_trainer rllib --use_s3" --start --stop \
##--cluster-name=ev_i210_test2 --tmux

# 5/10
#ray exec ray_autoscale.yaml "python flow/examples/train.py multiagent_straight_road \
#straight_road_reroute_local_rew_mpg --algorithm PPO \
#--num_iterations 200 --num_cpus 8 --num_rollouts 8 --rl_trainer rllib --use_s3" --start --stop \
#--cluster-name=ev_i210_test1 --tmux
#
#ray exec ray_autoscale.yaml "python flow/examples/train.py multiagent_i210 \
#i210_reroute_local_rew_mpg --algorithm PPO \
#--num_iterations 200 --num_cpus 8 --num_rollouts 8 --rl_trainer rllib --use_s3" --start --stop \
#--cluster-name=ev_i210_test2 --tmux

ray exec ray_autoscale.yaml "python flow/examples/train.py multiagent_straight_road \
straight_road_reroute_local_rew_mpg_curr --algorithm PPO \
--num_iterations 200 --num_cpus 7 --num_rollouts 7 --rl_trainer rllib --use_s3 --grid_search" --start --stop \
--cluster-name=ev_i210_test3 --tmux

ray exec ray_autoscale.yaml "python flow/examples/train.py multiagent_i210 \
i210_reroute_local_rew_mpg_curr --algorithm PPO \
--num_iterations 200 --num_cpus 7 --num_rollouts 7 --rl_trainer rllib --use_s3 --grid_search" --start --stop \
--cluster-name=ev_i210_test4 --tmux