#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py test --num_iters 2 --checkpoint_freq 1 \
#    --horizon 400" --start --stop --cluster-name test --tmux
ray exec ray_autoscale.yaml \
"python flow/examples/rllib/velocity_bottleneck.py test --num_iters 2 --checkpoint_freq 1 \
    --horizon 400 --use_s3" --start --stop --cluster-name test