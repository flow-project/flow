date="$(date +'%H%M%S')"
echo $date

function run_exp () {
    eval "ray exec scripts/ray_autoscale.yaml \"python ~/flow/multi_ring_transfer/ring_transfer_exp.py --use_s3 --multi_node --num_cpus 14 --num_rollouts 28 --num_iter 401 --checkpoint_freq 50 --exp_title $1 $2\" --start --stop --tmux --cluster-name kp_$date\_$1"
}
    

run_exp sugiyama "--num_av 1"
# run_exp 1_av_500 "--ring_length 500 --num_total_veh 44"
# run_exp 2_av_500 "--ring_length 500 --num_total_veh 44 --num_av 2" 
# run_exp 2_lanes_1_av "--num_lanes 2 --num_total_veh 44"
# run_exp 2_lanes_2_av "--num_lanes 2 --num_total_veh 44 --num_av 2"
# run_exp 2_lanes_2_av_500 "--num_lanes 2 --num_total_veh 88 --num_av 2"
# run_exp 4_lanes_4_av "--num_lanes 4 --num_total_veh 88 --num_av 4"
