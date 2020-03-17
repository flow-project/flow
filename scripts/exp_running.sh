#!/usr/bin/env bash

# 3/11 exps
ray exec ray_autoscale.yaml \
"python flow/examples/train.py multiagent_i210_b10000 --rl_trainer RLlib --num_cpus 1 --algorithm TD3 --exp_title test_td3 --use_s3" \
--start --stop --cluster-name ev_test_mem2 --tmux