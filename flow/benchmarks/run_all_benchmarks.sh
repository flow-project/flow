#!/bin/bash
echo "Running all benchmarks"

continue() {
echo -n "Continue executing? N will exit the run script. (Y/N)? "
while read -r -n 1 -s answer; do
  if [[ $answer = [YyNn] ]]; then
    [[ $answer = [Yy] ]] && retval=0
    [[ $answer = [Nn] ]] && retval=1
    break
  fi
done

echo # just a final linefeed, optics...

return $retval
}

declare -a benchmarks=("bottleneck0" "bottleneck1" "bottleneck2"
#                        "figureeight0" "figureeight1" "figureeight2"
#                        "figureeight0" "figureeight1" "figureeight2"
#                        "grid0" "grid1"
#                        "merge0" "merge1"
                        )
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$parent_path"

dt=$(date '+%Y_%m_%d_%H%M');
echo $dt
for run_script in rllib/*_runner.py; do
    for benchmark in "${benchmarks[@]}"; do
#        if continue(${benchmark}, ${run_script}); then
        echo "====================================================================="
        echo "Training ${benchmark} with ${run_script}"
        echo "====================================================================="
        ray exec ../../scripts/benchmark_autoscale.yaml "python ./flow/benchmarks/rllib/${run_script} --benchmark_name=${benchmark}" --start --stop --cluster-name=all_benchmark_${benchmark}_$dt
#        else
#            break
#        fi
    done
done