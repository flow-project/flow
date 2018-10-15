<img src="https://user-images.githubusercontent.com/38776306/44690767-e13efb80-aa10-11e8-9647-60933b1bc41a.png" align="right" width="400"/> 

# Flow 

[![Build Status](https://travis-ci.com/flow-project/flow.svg?branch=master)](https://travis-ci.com/flow-project/flow)
[![Docs](https://readthedocs.org/projects/flow/badge)](http://flow.readthedocs.org/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/flow-project/flow/blob/master/LICENSE.md)

[Flow](https://flow-project.github.io/) is a computational framework for deep RL and control experiments for traffic microsimulation. It provides a suite of traffic control scenarios (benchmarks), tools for designing custom traffic scenarios, and integration with deep reinforcement learning and traffic microsimulation libraries.

See [our website](https://flow-project.github.io/) for more information on the application of Flow to several mixed-autonomy traffic scenarios. Other [results and videos](https://sites.google.com/view/ieee-tro-flow/home) are available as well.

## Setup
1. Make sure you have Python 3 installed (we recommend using the [Anaconda Python distribution](https://www.continuum.io/downloads)).
2. **Install Jupyter** with `pip install jupyter`. Verify that you can start a Jupyter notebook with the command `jupyter-notebook`.
3. **Install Flow** by executing the following [installation instructions](https://berkeleyflow.readthedocs.io/en/latest/flow_setup.html).
  
## Documentation and Tutorials
  
Full documentation is [available online](https://flow.readthedocs.org/en/latest/).

[Tutorials](https://github.com/flow-project/flow/tree/master/tutorials): In addition to our documentation, we have also developed a set of tutorials which outline all the major features of Flow and its integrations with deep RL and traffic simulation libraries.

## Experiments and Results

### Phantom shockwave dissipation on a ring
Inspired by the famous [2008 Sugiyama experiment](https://www.youtube.com/watch?v=7wm-pZp_mi0) demonstrating spontaneous formation of traffic shockwaves (reproduced on the left video), and a [2017 field study](https://www.youtube.com/watch?v=2mBjYZTeaTc) demonstrating the ability of AVs to suppress shockwaves, we investigated the ability of reinforcement learning to train an optimal shockwave dissipating controller.

In the right video, we learn a controller (policy) for one out of 22 vehicles. By training on ring roads of varying lengths, and using a neural network policy with memory, we were able to learn a controller that both was optimal (in terms of average system velocity) and generalized outside of the training distribution.

[![Sugiyama simulation](https://img.youtube.com/vi/Lggtw9AOH0A/0.jpg =300x200)](https://www.youtube.com/watch?v=Lggtw9AOH0A)

[![Flow controller on ring road](https://img.youtube.com/vi/D0lNWWK3s9s/0.jpg =300x200)](https://www.youtube.com/watch?v=D0lNWWK3s9s)

### Intersection control
We demonstrated also the ability of a single autonomous vehicle to control the relative spacing of vehicles following behind it to create an optimal merge at an intersection.

As can be seen in the videos, without any AVs, the vehicles are stopped at the intersection by vehicles in the other direction; we show that even at low penetration rates, the autonomous vehicle "bunches" all of the other vehicles to avoid the intersection, resulting in a huge speed improvement.

[![Figure 8](https://img.youtube.com/vi/Z6QltFAEDeQ/0.jpg)](https://www.youtube.com/watch?v=Z6QltFAEDeQ)

[![Flow controller on figure 8](https://img.youtube.com/vi/SoA_7fPJEG8/0.jpg)](https://www.youtube.com/watch?v=SoA_7fPJEG8)

### Bottleneck control
Inspired by the rapid decrease in lanes on the San Francisco-Oakland Bay Bridge, we study a bottleneck that merges from four lanes down to two to one.

We demonstrate that the AVs are able to learn a strategy that increases the effective outflow at high inflows, and performs competitively with ramp metering.

<img src="https://flow-project.github.io/figures/experiments/bottleneck_control.png"/>

<img src="https://flow-project.github.io/figures/experiments/uncontrolVRLBigAxes.png"/>

## Getting Involved

We welcome your contributions.

- Ask questions on our mailing list: [flow-dev@googlegroups.com](https://groups.google.com/forum/#!forum/flow-dev).
- Please report bugs by submitting a [GitHub issue](https://github.com/flow-project/flow/issues).
- Submit contributions using [pull requests](https://github.com/flow-project/flow/pulls).

## Publications

### Citing Flow

If you use Flow for academic research, you are highly encouraged to cite our paper:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol. abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465

### Other Publications

Below are several other relevant publications based on our work:

- C. Wu, A. Kreidieh, E. Vinitsky, A. Bayen, "Emergent behaviors in mixed-autonomy traffic," Proceedings of the 1st Annual Conference on Robot Learning, PMLR 78:398-407, 2017. [Online]. Available: http://proceedings.mlr.press/v78/wu17a.html
- C. Wu, K. Parvate, N. Kheterpal, L. Dickstein, A. Mehta, E. Vinitsky, A. Bayen, "Framework for Control and Deep Reinforcement Learning in Traffic," IEEE Intelligent Transportation Systems Conference (ITSC), 2017. [Online]. Available: https://ieeexplore.ieee.org/document/8317694/
- E. Vinitsky, K. Parvate, A. Kreidieh, C. Wu, Z. Hu, A. Bayen, "Lagrangian Control through Deep-RL: Applications to Bottleneck Decongestion," IEEE Intelligent Transportation Systems Conference (ITSC), 2018.
- N. Kheterpal, K. Parvate, C. Wu, A. Kreidieh, E. Vinitsky, A. Bayen, "Flow: Deep Reinforcement for Control in SUMO," SUMO User Conference, 2018. [Online]. Available: https://easychair.org/publications/paper/FBQq
- A. Kreidieh, A. Bayen, "Dissipating stop-and-go waves in closed and open networks via deep reinforcement learning," IEEE Intelligent Transportation Systems Conference (ITSC), 2018.

## Contributors

Cathy Wu, Eugene Vinitsky, Aboudy Kreidieh, Marsalis Gibson, Fangyu Wu, Lucas Fischer, Crystal Yan, Kaila Cappello, Umah Sharaf, Xiao Zhao, Kathy Jang, Kanaad Parvate, Nishant Kheterpal, Ethan Hu, Kevin Chien, Jonathan Lin, Mahesh Murag. Alumni contributors include Saleh Albeaik, Ananth Kuchibhotla, Leah Dickstein and Nathan Mandi. 

Flow is supported by the [Mobile Sensing Lab](http://bayen.eecs.berkeley.edu/) at UC Berkeley (advised by Professor Alexandre Bayen) and Amazon AWS Machine Learning research grants.

Learn more about the team [here](https://flow-project.github.io/team.html).
