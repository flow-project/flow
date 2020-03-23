#!/usr/bin/env bash

# 3/11 exp
#ray exec ray_autoscale.yaml \
#"python flow/examples/train.py multiagent_i210 --rl_trainer RLlib --num_cpus 1 --algorithm TD3 --exp_title test_td3_b20000_h2000_n0p6_warm500 --use_s3" \
#--start --stop --cluster-name ev_test_mem5 --tmux

# 3/23 experiment
#ray exec ray_autoscale.yaml \
#"python flow/examples/train.py multiagent_i210 --rl_trainer RLlib --num_cpus 1 --algorithm TD3 --exp_title test_td3_b20000_h2000_n0p6 --use_s3 \
#--num_iterations 300" \
#--start --stop --cluster-name ev_test_mem5 --tmux

ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210 --rl_trainer RLlib --num_cpus 1 --algorithm TD3 --exp_title test_td3_gridsearch --use_s3 \
--num_iterations 300 --grid_search --n_cpus 15" \
--start --stop --cluster-name ev_test_mem6 --tmux