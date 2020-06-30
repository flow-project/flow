# Flow Examples

Before continuing to the Flow examples, we recommend **installing Flow** by 
following the [installation instructions](
https://flow.readthedocs.io/en/latest/flow_setup.html).

The **examples** folder provides several examples demonstrating how 
both non-RL simulation and RL-oriented simulatons can be setup and executed 
within the Flow framework on a variety of traffic problems. These examples are 
python files that may be executed either from terminal or via a text editor. 
For example, in order to execute the non-RL Ring example we run:

```shell script
python simulate.py "ring"
```

The examples are categorized into the following 3 sections:

## non-RL examples 

These are examples of transportation network with vehicles
following human-dynamical models of driving behavior using the traffic 
micro-simulator sumo and traffic macro-simulator Aimsun.

To execute these examples, run

```shell script
python simulate.py EXP_CONFIG 
```
where `EXP_CONFIG` is the name of the experiment configuration file, as located
in `exp_configs/non_rl.`

There are several *optional* arguments that can be added to the above command:

```shell script
 python simulate.py EXP_CONFIG --num_runs n --no_render --aimsun --gen_emission
```
where `--num_runs` indicates the number of simulations to run (default of `n` 
is 1), `--no_render` indicates whether to deactivate the simulation GUI during 
runtime (by default simulation GUI is active), `--aimsun` indicates whether to 
run the simulation using the simulator Aimsun (the default simulator is SUMO), 
and `--gen_emission` indicates whether to generate an emission file from the 
simulation.

## RL examples 

### RLlib

These examples are similar networks as those mentioned in *non-RL examples*, 
but in the presence of autonomous vehicle (AV) or traffic light agents being 
trained through RL algorithms provided by *RLlib*.

To execute these examples, run

```shell script
python train.py EXP_CONFIG --rl_trainer "rllib"
```
where `EXP_CONFIG` is the name of the experiment configuration file, as located
in `exp_configs/rl/singleagent` or  `exp_configs/rl/multiagent.`


### stable-baselines

These examples provide similar networks as those 
mentioned in *non-RL examples*, but in the presence of autonomous vehicle (AV) 
or traffic light agents being trained through RL algorithms provided by OpenAI 
*stable-baselines*.

To execute these examples, run

```shell script
python train.py EXP_CONFIG --rl_trainer "stable-baselines"
```
where `EXP_CONFIG` is the name of the experiment configuration file, as located
in `exp_configs/rl/singleagent.`

Note that, currently, multiagent experiments are only supported through RLlib.

There are several *optional* arguments that can be added to the above command:

```shell script
python train.py EXP_CONFIG --rl_trainer "stable-baselines" --num_cpus n1 --num_steps n2 --rollout_size r
```
where `--num_cpus` indicates the number of CPUs to use (default of `n1` is 1), 
`--num_steps` indicates the total steps to perform the learning (default of 
`n2` is 5000), and `--rollout_size` indicates the number of steps in a training
batch (default of `r` is 1000)

### h-baselines

A third RL algorithms package supported by the `train.py` script is 
[h-baselines](https://github.com/AboudyKreidieh/h-baselines). In order to use 
the algorithms supported by this package, begin by installing h-baselines by 
following the setup instructions located 
[here](https://flow.readthedocs.io/en/latest/flow_setup.html#optional-install-h-baselines). 
A policy can be trained using one of the exp_configs as follows:

```shell script
python examples/train.py singleagent_ring --rl_trainer h-baselines
```

**Logging:**

The above script executes a training operation and begins logging training and 
testing data under the path: *training_data/singleagent_ring/<date_time>*.

To visualize the statistics of various tensorflow operations in tensorboard, 
type:

```shell script
tensorboard --logdir <path/to/flow>/examples/training_data/singleagent_ring/<date_time>
```

Moreover, as training progressive, per-iteration and cumulative statistics are 
printed as a table on your terminal. These statistics are stored under the csv 
files *train.csv* and *eval.csv* (if also using an evaluation environment) 
within the same directory.

**Hyperparameters:**

When using h-baseline, multiple new command-line arguments can be passed to 
adjust the choice of algorithm and variable hyperparameters of the algorithms. 
These new arguments are as follows:

* `--alg` (*str*): The algorithm to use. Must be one of [TD3, SAC]. Defaults to
  'TD3'.
* `--evaluate` (*store_true*): whether to add an evaluation environment. The 
  evaluation environment is similar to the training environment, but with 
  `env_params.evaluate` set to True.
* `--n_training` (*int*): Number of training operations to perform. Each 
  training operation is performed on a new seed. Defaults to 1.
* `--total_steps` (*int*): Total number of timesteps used during training. 
  Defaults to 1000000.
* `--seed` (*int*): Sets the seed for numpy, tensorflow, and random. Defaults 
  to 1.
* `--log_interval` (*int*): the number of training steps before logging 
  training results. Defaults to 2000.
* `--eval_interval` (*int*): number of simulation steps in the training 
  environment before an evaluation is performed. Only relevant if `--evaluate` 
  is called. Defaults to 50000.
* `--save_interval` (int): number of simulation steps in the training 
  environment before the model is saved. Defaults to 50000.
* `--initial_exploration_steps` (*int*): number of timesteps that the policy is
  run before training to initialize the replay buffer with samples. Defaults to
  10000.
* `--nb_train_steps` (*int*): the number of training steps. Defaults to 1.
* `--nb_rollout_steps` (*int*): the number of rollout steps. Defaults to 1.
* `--nb_eval_episodes` (*int*): the number of evaluation episodes. Only 
  relevant if `--evaluate` is called. Defaults to 50.
* `--reward_scale` (*float*): the value the reward should be scaled by. 
  Defaults to 1.
* `--buffer_size` (*int*): the max number of transitions to store. Defaults to 
  200000.
* `--batch_size` (*int*): the size of the batch for learning the policy. 
  Defaults to 128.
* `--actor_lr` (*float*): the actor learning rate. Defaults to 3e-4.
* `--critic_lr` (*float*): the critic learning rate. Defaults to 3e-4.
* `--tau` (*float*): the soft update coefficient (keep old values, between 0 
  and 1). Defatuls to 0.005.
* `--gamma` (*float*): the discount rate. Defaults to 0.99.
* `--layer_norm` (*store_true*): enable layer normalisation
* `--use_huber` (*store_true*): specifies whether to use the huber distance 
  function as the loss for the critic. If set to False, the mean-squared error 
  metric is used instead")
* `--actor_update_freq` (*int*): number of training steps per actor policy 
  update step. The critic policy is updated every training step. Only used when 
  the algorithm is set to "TD3". Defaults to 2.
* `--noise` (*float*): scaling term to the range of the action space, that is 
  subsequently used as the standard deviation of Gaussian noise added to the 
  action if `apply_noise` is set to True in `get_action`". Only used when the 
  algorithm is set to "TD3". Defaults to 0.1.
* `--target_policy_noise` (*float*): standard deviation term to the noise from 
  the output of the target actor policy. See TD3 paper for more. Only used when
  the algorithm is set to "TD3". Defaults to 0.2.
* `--target_noise_clip` (*float*): clipping term for the noise injected in the
  target actor policy. Only used when the algorithm is set to "TD3". Defaults 
  to 0.5.
* `--target_entropy` (*float*): target entropy used when learning the entropy 
  coefficient. If set to None, a heuristic value is used. Only used when the 
  algorithm is set to "SAC". Defaults to None.

Additionally, the following arguments can be passed when training a multiagent 
policy:

* `--shared` (*store_true*): whether to use a shared policy for all agents
* `--maddpg` (*store_true*): whether to use an algorithm-specific variant of 
  the MADDPG algorithm


## Simulated Examples

The following networks are available for simulation within flow. These examples are 
all available as non-RL examples, while some of them are also available (with 
trainable variants) as RL examples, with RLlib or Stable Baselines.


### bay_bridge.py \& bay_bridge_toll.py

Perform simulations of vehicles on the Oakland-San Francisco Bay Bridge.

Unlike `bay_bridge.py`, `bay_bridge_toll.py` consists of vehicles being placed 
only on the toll booth and sections of the road leading up to it.

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/bay_bridge.gif)

### bottleneck.py

Example demonstrating formation of congestion in bottleneck

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/bottlenecks.gif)

### figure_eight.py

Example of a figure 8 network with human-driven vehicles.

Right-of-way dynamics near the intersection causes vehicles to queue up on
either side of the intersection, leading to a significant reduction in the
average speed of vehicles in the network.

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/figure_eight.gif)

### traffic_light_grid.py

Performs a simulation of vehicles on a traffic light grid.

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/grid.gif)

### highway.py

Example of an open multi-lane network with human-driven vehicles.

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/highway.gif)

### highway_ramps.py

Example of a highway section network with on/off ramps

![](picture to be added)

### merge.py

Example of a straight road with an on-ramp merge.

In the absence of autonomous vehicles, the network exhibits properties of
convective instability, with perturbations propagating upstream from the merge
point before exiting the network.

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/merge.gif)

### minicity.py

Example of modified mini city developed under a 
[collaboration with University of Delaware](https://sites.google.com/view/iccps-policy-transfer),
with human-driven vehicles.

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/minicity.gif)

### ring.py

Used as an example of a ring experiment.

This example consists of 22 IDM cars driving on a ring road creating shockwaves.

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/sugiyama.gif)
