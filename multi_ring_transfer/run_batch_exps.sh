date="$(date +'%H%M%S')"
echo $date

function run_exp () {
    eval "ray exec scripts/ray_autoscale.yaml \"python ~/flow/multi_ring_transfer/ring_transfer_exp.py --use_s3 --multi_node --num_cpus 14 --num_rollouts 28 --num_iter 401 --checkpoint_freq 50 --exp_title $1 $2\" --start --stop --tmux --cluster-name kp_$date\_$1"
}

run_exp 1_av_1_lane_220m "--horizon 1000 --num_av 1 --num_lanes 1 --ring_length 220 --num_veh 11"
run_exp 2_av_2_lane_220m "--horizon 1000 --num_av 2 --num_lanes 2 --ring_length 220 --num_veh 22"
run_exp 1_av_1_lane_440m "--horizon 1000 --num_av 1 --num_lanes 1 --ring_length 440 --num_veh 22"
# run_exp 1_av_1_lane_880m "--horizon 1000 --num_av 1 --num_lanes 1 --ring_length 880 --num_veh 44"
run_exp 2_av_1_lane_440m "--horizon 1000 --num_av 2 --num_lanes 1 --ring_length 440 --num_veh 22"
# run_exp 4_av_1_lane_880m "--horizon 1000 --num_av 4 --num_lanes 1 --ring_length 880 --num_veh 44"
run_exp 2_av_1_lane_220m "--horizon 1000 --num_av 2 --num_lanes 1 --ring_length 220 --num_veh 11"
run_exp 4_av_2_lane_220m "--horizon 1000 --num_av 4 --num_lanes 2 --ring_length 220 --num_veh 22"
run_exp 4_av_1_lane_440m "--horizon 1000 --num_av 4 --num_lanes 1 --ring_length 440 --num_veh 22"
# run_exp 8_av_1_lane_880m "--horizon 1000 --num_av 8 --num_lanes 1 --ring_length 880 --num_veh 44"
