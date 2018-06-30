# Flow Examples

Before continuing in this folder, we recommend **installing Flow** by executing 
the following [installation instructions](
https://berkeleyflow.readthedocs.io/en/latest/flow_setup.html).

The **examples** folder provides several examples demonstrating how 
both simulation and RL-oriented experiments can be setup and executed within 
the Flow framework on a variety of traffic problems. These examples are .py 
files that may be executed  either from terminal or via an editor. For example,
in order to execute the  sugiyama example in *examples/sumo*, we run:

```shell
python <flow-path>/examples/sumo/sugiyama.py
```

The examples are distributed into the following sections:

**examples/sumo/** contains examples of transportation network with vehicles
following human-dynamical models of driving behavior.

**examples/rllib/** provides similar networks as those presented in the 
previous point, but in the present of autonomous vehicle (AV) or traffic light 
agents being trained through RL algorithms provided by *RLlib*.

**examples/rllab/** provides similar examples as the one above, but where the 
RL agents are controlled and training the RL library *rllab*. Before running 
any experiment here, be sure to run `source activate flow`.
