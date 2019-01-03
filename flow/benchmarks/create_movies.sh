#!/bin/bash
echo "Preparing to make and save movies"

##################################################
# CONSTANTS
PERF_GUARANTEE=.95
##################################################

# line to evaluate non-integer divisions
calc(){ awk "BEGIN { print "$*" }"; }

# save this so we can find visualizer_rllib.py
script_path=$(pwd)

# source shflags
. $script_path/../flow/utils/shflags

# define a 'name' command-line string flag
DEFINE_string 'filepath' 'no_flag' 'path to outer folder with pkl files' 'f'
DEFINE_boolean 'bmmode' true 'whether to check if benchmarks satisfy metrics' 'b'

# parse the command-line
FLAGS "$@" || exit $?
eval set -- "${FLAGS_ARGV}"

# check if a path to the checkpoints was passed
file_path=
if [ "${FLAGS_filepath}" == 'no_path' ]; then
    echo "Please pass the path to the outer folder containing benchmarks"
    exit 1
fi

echo "entering data folder: ${FLAGS_filepath}"
cd "${FLAGS_filepath}"

# create an array containing the benchmark names and expected metrics
declare -a benchmarks=(
                        "bottleneck0" "bottleneck1" "bottleneck2"
                        "figureeight0" "figureeight1" "figureeight2"
                        "grid0" "grid1"
                        "merge0" "merge1" "merge2"
                        )

# step into every pulled folder
for outer_folder in */; do
    cd $outer_folder
    for inner_folder in */; do
        cd $inner_folder
        checkpoint_num="$(ls | grep '^checkpoint_[0-9]\+$' | cut -c12- | sort -n | tail -n1)"
        echo "====================================================================="
        echo "Visualizing most recent checkpoint in "$outer_folder
        echo "====================================================================="
        file_path=$(pwd)

        # if you want to evaluate the benchmarks
        if [ ${FLAGS_bmmode} -eq ${FLAGS_TRUE} ]; then
            python $script_path/../flow/visualize/visualizer_rllib.py $file_path $checkpoint_num \
            --save_render --num_rollouts 1 --horizon 1 > $script_path/tmp.txt
            # read out the text file to find the avg velocity and avg outflow
            speed_str=$(grep "Average, std speed" $script_path/tmp.txt)
            outflow_str=$(grep "Average, std outflow" $script_path/tmp.txt)
            rew_str=$(grep "Average, std return" $script_path/tmp.txt)
            # parse the speed, outflow, and reward from the string
            IFS=' ' read -ra speed_arr <<< "$speed_str"
            speed=${speed_arr[3]}
            speed=$(echo "${speed%?}")
            IFS=' ' read -ra outflow_arr <<< "$outflow_str"
            outflow=${outflow_arr[3]}
            outflow=$(echo "${outflow%?}")
            IFS=' ' read -ra rew_arr <<< "$rew_str"
            rew=${outflow_arr[3]}
            rew=$(echo "${rew%?}")
            unset IFS

            folder_name=$(echo "${outer_folder%?}")

            failed_exps="failed exps are: "
            fail_flag=0
            # now figure out whether benchmark should be evaluated by speed,
            # outflow, or reward
            # benchmarks with outflow rewards
            if [[ $folder_name == "bottleneck0" ]] || [[ $folder_name == "bottleneck1" ]] \
            || [[ $folder_name == "bottleneck2" ]]; then
                if [ $folder_name == "bottleneck0" ]; then
                    performance=$(calc $outflow/1167.0)
                    if [ $(echo $performance '<' $PERF_GUARANTEE | bc -l) == 1 ]; then
                        echo "bottleneck 0 underperformed"
                        failed_exps=$failed_exps"bottleneck0 "
                        fail_flag=1
                    fi
                elif [ $folder_name == "bottleneck1" ]; then
                    performance=$(calc $outflow/1258.0)
                    if [ $(echo $performance '<' $PERF_GUARANTEE | bc -l) == 1 ]; then
                        echo "bottleneck 1 underperformed"
                        failed_exps=$failed_exps"bottleneck1 "
                        fail_flag=1
                    fi
                elif [ $folder_name == "bottleneck2" ]; then
                    performance=$(calc $outflow/2134.0)
                    if [ $(echo $performance '<' $PERF_GUARANTEE | bc -l) == 1 ]; then
                        echo "bottleneck 2 underperformed"
                        failed_exps=$failed_exps"bottleneck2 "
                        fail_flag=1
                    fi
                fi

            # benchmarks with speed rewards
            elif [[ $folder_name == "figureeight0" ]] || [[ $folder_name == "figureeight1" ]] \
            || [[ $folder_name == "figureeight2" ]] || [[ $folder_name == "merge0" ]] \
            || [[ $folder_name == "merge1" ]] || [[ $folder_name == "merge2" ]] ; then
                if [ $folder_name == "figureeight0" ]; then
                    performance=$(calc $speed/7.3)
                    if [ $(echo $performance '<' $PERF_GUARANTEE | bc -l) == 1 ]; then
                        echo "figureeight 0 underperformed"
                        failed_exps=$failed_exps"figureeight0 "
                        fail_flag=1
                    fi
                elif [ $folder_name == "figureeight1" ]; then
                    performance=$(calc $speed/6.4)
                    if [ $(echo $performance '<' $PERF_GUARANTEE | bc -l) == 1 ]; then
                        echo "figureeight 1 underperformed"
                        failed_exps=$failed_exps"figureeight1 "
                        fail_flag=1
                    fi
                elif [ $folder_name == "figureeight2" ]; then
                    performance=$(calc $speed/5.7)
                    if [ $(echo $performance '<' $PERF_GUARANTEE | bc -l) == 1 ]; then
                        echo "figureeight 2 underperformed"
                        failed_exps=$failed_exps"figureeight2 "
                        fail_flag=1
                    fi
                elif [ $folder_name == "merge0" ]; then
                    performance=$(calc $speed/13.0)
                    if [ $(echo $performance '<' $PERF_GUARANTEE | bc -l) == 1 ]; then
                        echo "merge 0 underperformed"
                        failed_exps=$failed_exps"merge0 "
                        fail_flag=1
                    fi
                elif [ $folder_name == "merge1" ]; then
                    performance=$(calc $speed/13.0)
                    if [ $(echo $performance '<' $PERF_GUARANTEE | bc -l) == 1 ]; then
                        echo "merge 1 underperformed"
                        failed_exps=$failed_exps"merge1 "
                        fail_flag=1
                    fi
                elif [ $folder_name == "merge2" ]; then
                    performance=$(calc $speed/13.0)
                    if [ $(echo $performance '<' $PERF_GUARANTEE | bc -l) == 1 ]; then
                        echo "merge 2 underperformed"
                        failed_exps=$failed_exps"merge2 "
                        fail_flag=1
                    fi
                fi
            # benchmarks that use the reward as their metric
            else
                if [ $folder_name == "grid0" ]; then
                    performance=$(calc $rew/296.0)
                    if [ $(echo $performance '<' 0.97 | bc -l) == 1 ]; then
                        echo "grid 0 underperformed"
                        failed_exps=$failed_exps"grid0 "
                        fail_flag=1
                    fi
                elif $folder_name == "grid1"; then
                    performance=$(calc $rew/296.0)
                    if [ $(echo $performance '<' 0.97 | bc -l) == 1 ]; then
                        echo "grid 1 underperformed"
                        failed_exps=$failed_exps"grid1 "
                        fail_flag=1
                    fi
                fi
            fi
            # remove the made file
            rm $script_path/tmp.txt
        else
            python $script_path/../flow/visualize/visualizer_rllib.py $file_path $checkpoint_num \
            --save_render
        fi
        cd "../"
    done
    cd "../"
done

if [[ ${FLAGS_bmmode} -eq ${FLAGS_TRUE} ]] && [[ $fail_flag=1 ]]; then
    echo $failed_exps
fi
