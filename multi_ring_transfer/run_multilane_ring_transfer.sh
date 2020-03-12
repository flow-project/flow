if [ -z "$1" ]
  then
    echo "Need to specify an experiment name."
    exit
fi

date="$(date +'%d_%m_%H%M%S')"
echo $date
while true; do
    read -p "Run experiment: $1 on cluster? " yn
    case $yn in
        [Yy]* ) ray exec scripts/ray_autoscale.yaml "python ~/flow/multi_ring_transfer/ring_transfer_exp.py \
                                     --use_s3 --multi_node \
                                     --exp_name $1_$date\
                                     --num_cpus 4 --num_rollouts 8 \
                                     --num_iter 1" --start --stop --cluster-name kp_$date_$1; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done