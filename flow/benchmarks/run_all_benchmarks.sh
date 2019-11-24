#!/bin/bash
echo "Running all benchmarks"

declare -a benchmarks=(
                        "bottleneck0" "bottleneck1" "bottleneck2"
                        "figureeight0" "figureeight1" "figureeight2"
                        "grid0" "grid1"
                        "merge0" "merge1"
                        )
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$parent_path"

dt=$(date '+%Y_%m_%d_%H%M');
echo $dt
i=0
for run_script in rllib/ppo_runner.py; do
    declare alg=`echo ${run_script} | cut -d'/' -f 2 | cut -d'_' -f 1`
    for benchmark in "${benchmarks[@]}"; do
        i=$((i+1))
        echo "====================================================================="
        echo "Training ${benchmark} with ${alg}"
        echo "ray exec ../../scripts/benchmark_autoscale.yaml \"python ./flow/flow/benchmarks/${run_script} --upload_dir=\"flow-benchmark.results/${dt}/\" --benchmark_name=${benchmark} --num_cpus 14\" --start --stop --cluster-name=all_benchmark_${benchmark}_${alg}_$dt --tmux"
        echo "====================================================================="
        ray exec ../../scripts/benchmark_autoscale.yaml "python ./flow/flow/benchmarks/${run_script} --upload_dir=\"flow-benchmark.results/${dt}/\" --benchmark_name=${benchmark} --num_cpus 14" --start --stop --cluster-name=all_benchmark_${benchmark}_${alg}_$dt --tmux
    done
done