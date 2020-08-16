# run cluster without training
ray exec ray_autoscale.yaml "echo OK" --start --cluster-name nathan-emissions --tmux

# local testing
python examples/train.py multiagent_i210 --exp_title i210-test --num_iterations 1 --num_rollouts 1 --render

#######################################################################################################
# 13/08/20 # 17h47 # Trainings with warm down, late penalty, accumulated reward # Instant fuel reward #
#######################################################################################################
# HORIZON = 1000 # WARMUP_STEPS = 600 # SIMS_PER_STEP = 3 # SIM_STEP = 0.4 ## PENETRATION_RATE = 0.10 #
#######################################################################################################

"""
TOTAL: 28 EXPS

Test combinations and tunings of:
- warm down (params: when it starts)
- late penalty (params: when it triggers and for how much)
- accumulated reward (params: how often reward is given and how much bonus)

vehicles seem to spend 300-400 seconds in the network once it's getting saturated
-> warm down around 300s, late penalty 600s (although they should be the same)
fuel rewards are around 0.01, 0.1 when summed over look back length of 5, 10
-> penalty of -1 seems adequate
-> accumulated reward bonus of interval/5 is enough to make it positive

/!\ These may have been wrong bc of some issues
- params not being saved into params.json for some reason (maybe because obtained from cli in train.py)
- rewards seem too high?
"""

# warm down + late penalty (12 exps)
for warm_down in 200 300 400 600
do
    for penalty in 600 700 800
    do
        echo ${warm_down} ${penalty}
        ray exec ray_autoscale.yaml \
        "python flow/examples/train.py multiagent_i210 --num_iterations 60 --num_rollouts 20 \
        --checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
        --warm_down --warm_down_steps ${warm_down} --look_back_length 5 \
        --late_penalty --late_penalty_steps ${penalty} --late_penalty_value -1 \
        --upload_graphs Nathan lxpj_i210_0p1_nrj_wd${warm_down}_pen${penalty}_lb5 \
        --exp_title lxpj_i210_0p1_nrj_wd${warm_down}_pen${penalty}_lb5" \
        --start --tmux --cluster-name nathan-lxpj_i210_0p1_nrj_wd${warm_down}_pen${penalty}_lb5
    done
done

# late penalty alone (4 exps)
for penalty in 500 600 700 800
do
    echo ${penalty}
    ray exec ray_autoscale.yaml \
    "python flow/examples/train.py multiagent_i210 --num_iterations 60 --num_rollouts 20 \
    --checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
    --look_back_length 5 \
    --late_penalty --late_penalty_steps ${penalty} --late_penalty_value -1 \
    --upload_graphs Nathan y8xm_i210_0p1_nrj_pen${penalty}_lb5 \
    --exp_title y8xm_i210_0p1_nrj_pen${penalty}_lb5" \
    --start --tmux --cluster-name nathan-y8xm_i210_0p1_nrj_pen${penalty}_lb5
done

# accumulated reward alone (5 exps)
for rwd_interval in 1 5 10 20 100
do
    echo ${rwd_interval} $(echo "${rwd_interval}/3" | bc -l)
    ray exec ray_autoscale.yaml \
    "python flow/examples/train.py multiagent_i210 --num_iterations 60 --num_rollouts 20 \
    --checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
    --look_back_length 5 \
    --accumulated_reward --accumulated_reward_interval_steps ${rwd_interval} --accumulated_reward_bonus $(echo "${rwd_interval}/3" | bc -l) \
    --upload_graphs Nathan jt0j_i210_0p1_nrj_accr${rwd_interval}_lb5 \
    --exp_title jt0j_i210_0p1_nrj_accr${rwd_interval}_lb5" \
    --start --tmux --cluster-name nathan-jt0j_i210_0p1_nrj_accr${rwd_interval}_lb5
done

# warm down + late penalty + accumulated reward (4 exps)
for warm_down in 500 700
do
    for penalty in 500 700
    do
        for rwd_interval in 20
        do
            echo ${warm_down} ${penalty} ${rwd_interval} $(echo "${rwd_interval}/3" | bc -l)
            ray exec ray_autoscale.yaml \
            "python flow/examples/train.py multiagent_i210 --num_iterations 60 --num_rollouts 20 \
            --checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
            --warm_down --warm_down_steps ${warm_down} --look_back_length 5 \
            --late_penalty --late_penalty_steps ${penalty} --late_penalty_value -1 \
            --accumulated_reward --accumulated_reward_interval_steps ${rwd_interval} --accumulated_reward_bonus $(echo "${rwd_interval}/3" | bc -l) \
            --upload_graphs Nathan qnck_i210_0p1_nrj_wd${warm_down}_pen${penalty}_accr${rwd_interval}_lb5 \
            --exp_title qnck_i210_0p1_nrj_wd${warm_down}_pen${penalty}_accr${rwd_interval}_lb5" \
            --start --tmux --cluster-name nathan-qnck_i210_0p1_nrj_wd${warm_down}_pen${penalty}_accr${rwd_interval}_lb5
        done
    done
done

# fixed warm down and penalty with different look back lengths (3 exps)
for warm_down in 600
do
    for penalty in 600
    do
        for look_back_length in 1 6 11
        do
            echo ${warm_down} ${penalty} ${look_back_length}
            ray exec ray_autoscale.yaml \
            "python flow/examples/train.py multiagent_i210 --num_iterations 60 --num_rollouts 20 \
            --checkpoint_freq 10 --use_s3 --num_cpus 17 --algorithm PPO --rl_trainer rllib --grid_search --multi_node \
            --warm_down --warm_down_steps ${warm_down} --look_back_length ${look_back_length} \
            --late_penalty --late_penalty_steps ${penalty} --late_penalty_value -1 \
            --upload_graphs Nathan rotm_i210_0p1_nrj_wd${warm_down}_pen${penalty}_lb${look_back_length} \
            --exp_title rotm_i210_0p1_nrj_wd${warm_down}_pen${penalty}_lb${look_back_length}" \
            --start --tmux --cluster-name nathan-rotm_i210_0p1_nrj_wd${warm_down}_pen${penalty}_lb${look_back_length}
        done
    done
done
