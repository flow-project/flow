#!/bin/bash

#ray exec ray_autoscale.yaml "python flow/examples/train.py multiagent_i210 i210_reroute_test --algorithm PPO \
#--num_iterations 200 --num_cpus 12 --num_rollouts 12 --rl_trainer rllib --use_s3" --start --stop \
#--cluster-name=ev_i210_test --tmux

ray exec ray_autoscale.yaml "python flow/examples/train.py multiagent_i210 i210_reroute_test2 --algorithm PPO \
--num_iterations 200 --num_cpus 4 --num_rollouts 4 --rl_trainer rllib --use_s3" --start --stop \
--cluster-name=ev_i210_test2 --tmux