#!/usr/bin/env bash

# 3/11 exps
ray exec ray_autoscale.yaml \
"python flow/examples/train.py --exp_title imitation_test --num_iterations 20 --num_rollouts 10 --num_cpus 10 \
--imitate" \
--start --stop --cluster-name ev_test_im