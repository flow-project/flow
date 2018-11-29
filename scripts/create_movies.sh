#!/bin/bash
echo "Preparing to make and save movies"

script_path=$(pwd)

file_path=
if [ $# == 0 ]; then
    echo "Please pass the path to the outer folder containing benchmarks"
    exit 1
else
    file_path=$1
fi
cd "$file_path"

# step into every pulled folder
for outer_folder in */; do
    cd $outer_folder
    for inner_folder in */; do
        cd $inner_folder
        checkpoint_num="$(ls | grep '^checkpoint_[0-9]\+$' | cut -c12- | sort -n | tail -n1)"
        echo "====================================================================="
        echo "Visualizing highest checkpoints in "$outer_folder
        echo "====================================================================="
        file_path=$(pwd)
        python $script_path/../flow/visualize/visualizer_rllib.py $file_path $checkpoint_num \
        --save_render
        cd "../"
    done
    cd "../"
done
