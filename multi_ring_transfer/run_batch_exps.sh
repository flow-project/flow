date="$(date +'%H%M%S')"
echo $date

ray exec scripts/ray_autoscale.yaml "python ~/flow/multi_ring_transfer/ring_transfer_exp.py --use_s3 --multi_node --num_cpus 14 --num_rollouts 28 --num_iter 401 --checkpoint_freq 50 \
                                     --exp_name $1_$date
                                     " --start --stop --tmux --cluster-name kp_$date\_$1