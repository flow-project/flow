#!/bin/bash
echo "Visualizing all benchmarks \n"

echo "Please pass the pass to the folder containing the benchmark pkl files" \
"And hit [ENTER] \n"
read file_path

cd "$file_path"
# step into every pulled folder
for outer_folder in ./; do
    for inner_folder in ./; do
    # find the number of the highest checkpoint
    list=$(find . -name 'checkpoint*' -maxdepth 1 | sort -n)
    file_name=${list[-1]}
    # get the checkpoint number as we will need to pass it to visualizer
    checkpoint_num="$(cut -d'_' -f2 <<<"$file_name")"
    file_path=$(pwd)$file_name
    python ./../flow/visualize/visualizer_rllib.py $file_path $checkpoint_num \
    --save_render
    done
done