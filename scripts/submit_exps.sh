#!/bin/bash

ray submit ray_autoscale.yaml --start --stop --tmux --cluster-name=EugeneMultiAgentLCAggComm ../examples/rllib/multiagent_exps/MA_bottle_lc_agg_comm.py
ray submit ray_autoscale.yaml --start --stop --tmux --cluster-name=EugeneMultiAgentLCAggNoComm ../examples/rllib/multiagent_exps/MA_bottle_lc_agg_nocomm.py
ray submit ray_autoscale.yaml --start --stop --tmux --cluster-name=EugeneMultiAgentLCNoAggComm ../examples/rllib/multiagent_exps/MA_bottle_lc_noagg_comm.py
ray submit ray_autoscale.yaml --start --stop --tmux --cluster-name=EugeneMultiAgentLCNoAggNoComm ../examples/rllib/multiagent_exps/MA_bottle_lc_noagg_nocomm.py
ray submit ray_autoscale.yaml --start --stop --tmux --cluster-name=EugeneMultiAgentAggComm ../examples/rllib/multiagent_exps/MA_bottle_nolc_agg_comm.py
ray submit ray_autoscale.yaml --start --stop --tmux --cluster-name=EugeneMultiAgentAggNoComm ../examples/rllib/multiagent_exps/MA_bottle_nolc_agg_nocomm.py
ray submit ray_autoscale.yaml --start --stop --tmux --cluster-name=EugeneMultiAgentNoAggComm ../examples/rllib/multiagent_exps/MA_bottle_nolc_noagg_comm.py
ray submit ray_autoscale.yaml --start --stop --tmux --cluster-name=EugeneMultiAgentNoAggNoComm ../examples/rllib/multiagent_exps/MA_bottle_nolc_noagg_nocomm.py
ray submit ray_autoscale.yaml --start --stop --tmux --cluster-name=EugeneMultiAgentNoAggCommLSTM ../examples/rllib/multiagent_exps/MA_bottle_nolc_noagg_comm_lstm.py