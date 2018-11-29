#!/bin/bash
echo "Visualizing all benchmarks"

continue() {
echo -n "Continue executing? N will exit the run script. (Y/N)? "
while read -r -n 1 -s answer; do
  if [[ $answer = [YyNn] ]]; then
    [[ $answer = [Yy] ]] && retval=0
    [[ $answer = [Nn] ]] && retval=1
    break
  fi
done

continue() {
echo "Please pass the pass to the folder containing the benchmark pkl files" \
"And hit [ENTER]"
read file_path

cd "$file_path"
# step into every pulled folder
for outer_folder in ./; do
    for inner_folder in ./; do
    # find the number of the highest checkpoint
    list=$(find . -name 'checkpoint*' -maxdepth 1 | sort -n)
    file=${list[-1]}
    
    done
done

return $retval
}

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
for run_script in rllib/*_runner.py; do
    for benchmark in "${benchmarks[@]}"; do
        declare alg=`echo ${run_script} | cut -d'/' -f 2 | cut -d'_' -f 1`
        echo "====================================================================="
        echo "Training ${benchmark} with ${alg}"
        echo "ray exec ../../scripts/benchmark_autoscale.yaml \"python ./flow/flow/benchmarks/${run_script} --upload_dir=\"flow-benchmark.results/${dt}/\" --benchmark_name=${benchmark}\" --start --stop --cluster-name=all_benchmark_${benchmark}_${alg}_$dt --tmux"
        echo "====================================================================="
        ray exec ../../scripts/benchmark_autoscale.yaml "python ./flow/flow/benchmarks/${run_script} --upload_dir=\"flow-benchmark.results/${dt}/\" --benchmark_name=${benchmark}" --start --stop --cluster-name=all_benchmark_${benchmark}_${alg}_$dt --tmux
    done
done