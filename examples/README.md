# Flow Examples

Before continuing to the Flow examples, we recommend **installing Flow** by 
executing the following [installation instructions](
https://flow.readthedocs.io/en/latest/flow_setup.html).

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
following human-dynamical models of driving behavior using the traffic 
micro-simulator sumo.

**examples/aimsun/** contains examples of transportation network with vehicles
following human-dynamical models of driving behavior using the traffic 
micro-simulator Aimsun.

**examples/rllib/** provides similar networks as those presented in the 
previous point, but in the present of autonomous vehicle (AV) or traffic light 
agents being trained through RL algorithms provided by *RLlib*.


## Simulated Examples

The following networks are available for simulation within flow, and 
specifically the **examples/sumo** folder. Similar networks are available with 
trainable variants in the examples/rllib and examples/aimsun folders; however, 
they may be under different names.

### bay_bridge.py \& bay_bridge_toll.py

Perform simulations of vehicles on the Oakland-San Francisco Bay Bridge.

Unlike `bay_bridge.py`, `bay_bridge_toll.py` consists of vehicles being placed 
only on the toll booth and sections of the road leading up to it.

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/bay_bridge.gif)

### bottlenecks.py

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

### merge.py

Example of a merge network with human-driven vehicles.

In the absence of autonomous vehicles, the network exhibits properties of
convective instability, with perturbations propagating upstream from the merge
point before exiting the network.

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/merge.gif)

### minicity.py

Example of modified minicity of University of Delaware network with 
human-driven vehicles.

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/minicity.gif)

### sugiyama.py

Used as an example of sugiyama experiment.

This example consists of 22 IDM cars on a ring road creating shockwaves.

![](https://raw.githubusercontent.com/flow-project/flow/master/docs/img/sugiyama.gif)
