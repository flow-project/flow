# Flow Examples

Before continuing to the Flow examples, we recommend **installing Flow** by 
following the [installation instructions](
https://flow.readthedocs.io/en/latest/flow_setup.html).

The **examples** folder provides several examples demonstrating how 
both non-RL simulation and RL-oriented simulatons can be setup and executed 
within the Flow framework on a variety of traffic problems. These examples are 
python files that may be executed either from terminal or via a text editor. 
For example, in order to execute the non-RL Ring example we run:

```shell
python simulate.py ring
```

The examples are categorized into the following 3 sections:

## non-RL examples 

These are examples of transportation network with vehicles
following human-dynamical models of driving behavior using the traffic 
micro-simulator sumo and traffic macro-simulator Aimsun.

To execute these examples, run

```shell
python simulate.py EXP_CONFIG 
```
where `EXP_CONFIG` is the name of the experiment configuration file, as located in 
`exp_configs/non_rl.`

There are several *optional* arguments that can be added to the above command:

```shell
 python simulate.py EXP_CONFIG --num_runs n --no_render --aimsun --gen_emission
```
where `--num_runs` indicates the number of simulations to run (default of `n` is 1), `--no_render` indicates whether to deactivate the simulation GUI during runtime (by default simulation GUI is active), `--aimsun` indicates whether to run the simulation using the simulator Aimsun (the default simulator is SUMO), and `--gen_emission` indicates whether to generate an emission file from the simulation.

## RL examples based on RLlib 

These examples are similar networks as those mentioned in *non-RL examples*, but in the 
presence of autonomous vehicle (AV) or traffic light agents 
being trained through RL algorithms provided by *RLlib*.

To execute these examples, run

```shell
 python train.py EXP_CONFIG
 (or python train.py EXP_CONFIG --rl_trainer RLlib)
```
where `EXP_CONFIG` is the name of the experiment configuration file, as located in 
`exp_configs/rl/singleagent` or  `exp_configs/rl/multiagent.`


## RL examples based on stable_baselines 

These examples provide similar networks as those 
mentioned in *non-RL examples*, but in the presence of autonomous vehicle (AV) or traffic 
light agents being trained through RL algorithms provided by OpenAI *stable 
baselines*.

To execute these examples, run

```shell
 python train.py EXP_CONFIG --rl_trainer Stable-Baselines
```
where `EXP_CONFIG` is the name of the experiment configuration file, as located in 
`exp_configs/rl/singleagent.`

Note that, currently, multiagent experiments are only supported through RLlib.

There are several *optional* arguments that can be added to the above command:

```shell
 python train.py EXP_CONFIG --rl_trainer Stable-Baselines --num_cpus n1 --num_steps n2 --rollout_size r
```
where `--num_cpus` indicates the number of CPUs to use (default of `n1` is 1), `--num_steps` indicates the total steps to perform the learning (default of `n2` is 5000), and `--rollout_size` indicates the number of steps in a training batch (default of `r` is 1000)

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
