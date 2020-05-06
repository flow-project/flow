# Flow Benchmarks

This folder presents several several reinforcement learning benchmarks for 
training traffic lights and autonomous vehicles in a variety of mixed-autonomy 
traffic settings.

## Description of Benchmarks

For a detailed description of each benchmark, we refer the user to the Flow 
Benchmarks paper (see *Citing Flow Benchmarks*). At a higher level, the traffic
benchmarks presented in here are as follows:

**Figure eight (optimizing intersection capacity):** A portion of vehicles are 
treated as CAVs with the objective of regulating the flow of vehicles through 
the intersection of a figure eight in order to improve system-level velocities.
- `flow.benchmarks.figureeight0` 13 humans, 1 CAV, S=(28,), A=(1,), T=1500.
- `flow.benchmarks.figureeight1` 7 humans, 7 CAVs, S=(28,), A=(7,), T=1500.
- `flow.benchmarks.figureeight2` 0 human, 14 CAVs, S=(28,), A=(14,), T=1500.

**Merge (controlling shockwaves from on-ramp merges):** In a mixed-autonomy 
setting, a percentage of vehicles in the main highway of a merge network are 
tasked with the objective of dissipating the formation and propagation of 
*stop-and-go waves* from locally observable information.
- `flow.benchmarks.merge0` 10% CAV penetration rate, S=(25,), A=(5,), T=750.
- `flow.benchmarks.merge1` 25% CAV penetration rate, S=(65,), A=(13,), T=750.
- `flow.benchmarks.merge2` 33.3% CAV penetration rate, S=(85,), A=(17,), T=750.

**Traffic Light Grid (improving traffic signal timing schedules):** Traffic
lights in a an idealized representation of a city with a grid-like structure
such as Manhattan are controlled in intervals of 2 seconds, with the objective
of minimizing delays for drivers.
- `flow.benchmarks.grid0` 3x3 traffic light grid (9 traffic lights), 
inflow = 300 veh/hour/lane S=(339,), A=(9,), T=400.
- `flow.benchmarks.grid1` 5x5 traffic light grid (25 traffic lights), 
inflow = 300 veh/hour/lane S=(915,), A=(25,), T=400.

**Bottleneck (maximizing throughput in a bottleneck structure):** The goal of 
this problem is to learn to avoid the *capacity drop* that is characteristic to 
bottleneck structures in transportation networks, and maximize the total 
outflow in a mixed-autonomy setting. 
- `flow.benchmarks.bottleneck0` 4 lanes, inflow = 2500 veh/hour, 10% CAV
penetration, no vehicles are allowed to lane change, S=(141,), A=(20,), T=1000.
- `flow.benchmarks.bottleneck1` 4 lanes, inflow = 2500 veh/hour, 10% CAV
penetration, the human drivers follow the standard lane changing model in the 
simulator, S=(141,), A=(20,), T=1000.
- `flow.benchmarks.bottleneck2` 8 lanes, inflow = 5000 veh/hour, 10% CAV
penetration, no vehicles are allowed to lane change, S=(281,), A=(40,), T=1000.

## Training on Custom Algorithms

All benchmarks presented here are compatible with OpenAI gym in order to 
promote integration with the majority of training algorithms currently being 
developed by the RL community. The below code snippet presents a sample method
for importing and training the presented benchmarks on a dummy RL algorithm.
Techniques of running the benchmarks may vary from algorithm, however, and we 
provide specific executable implementations of the benchmarks on the RL 
library RLlib in the folder flow/benchmarks/rllib/.

```python
# import an RL algorithm for training
from foo import myAlgorithm

# import the experiment-specific parameters from flow.benchmarks
from flow.benchmarks.figureeight0 import flow_params

# import the make_create_env to register the environment with OpenAI gym
from flow.utils.registry import make_create_env

if __name__ == "__main__":
    # the make_create_env function produces a method that can be used to 
    # generate parameterizable gym environments that are compatible with Flow. 
    # This method will run both "register" and "make" (see gym documentation).
    # If these are supposed to be run within your algorithm/library, we 
    # recommend referring to the make_create_env source code in 
    # flow/utils/registry.py.
    create_env, env_name = make_create_env(flow_params, version=0)

    # create and register the environment with OpenAI Gym
    env = create_env()

    # setup the algorithm with the traffic environment. This may be either by 
    # specifying the environment's name or the env variable created by the 
    # create_env() method
    alg = myAlgorithm(env_name=env_name)
    alg.train()
```

## Citing Flow Benchmarks

If you use the following benchmarks for academic research, you are highly 
encouraged to cite our paper:

Vinitsky, E., Kreidieh, A., Le Flem, L., Kheterpal, N., Jang, K., Wu, F., ... & Bayen, A. M. (2018, October). Benchmarks for reinforcement learning in mixed-autonomy traffic. In Conference on Robot Learning (pp. 399-409).

## Evaluating benchmarks on EC2

The `run_all_benchmarks.sh` script will run each benchmark over all runners specified in the rllib folder on EC2,
allowing a user to quickly start instances that will validate their changes (serves as regression tests for Flow).
