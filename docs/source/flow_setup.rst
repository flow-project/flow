Setup Instructions
*****************************

To get flow running, you need three things: flow (or
flow), SUMO, and rllab. Once each component is installed successfully,
you might get some missing module bugs from python. Just install the
missing module using your OS-specific package manager / installation
tool. Follow the shell commands below to get started.

Installation (rllib version)
=================

Install [Anaconda](https://www.anaconda.com/download) for python and enable
it right away.
::

    source ~/.bashrc

Optionally create a conda environment named `flow`:
::

    conda create -n flow python=3.6 anaconda

Install flow
::

    git clone https://github.com/cathywu/flow.git
    cd flow
    bash scripts/setup_sumo_osx.sh <DESIRED_PATH_TO_SUMO>  # installs sumo at <DESIRED_PATH_TO_SUMO>/sumo
    python setup.py develop  # (install flow and dependencies)
    cp flow/core/config.template.py flow/core/config.py  # User config

Add the following to `~/.bashrc`
::

    export SUMO_HOME="<DESIRED_PATH_TO_SUMO>/sumo"
    export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
    export PATH="$SUMO_HOME/bin:$PATH"

Install ray
::

    git clone https://github.com/ray-project/ray.git
    pushd ray/python
    sudo apt-get install -y cmake
    python setup.py develop
    popd
    conda install opencv  # ray dependency


Test the installation
=====================

To run any of the examples, make sure to run
::
    source activate flow
    
Running the following should result in the loading of the SUMO GUI.
Click the run button and you should see unstable traffic form after a
few seconds, a la (Sugiyama et al, 2008).

Run the unit tests:

::

    nose2

Let’s see some traffic action:

::

    python examples/sugiyama.py

This means that you have Flow properly configured with SUMO.

::

    python examples/mixed_rl_single_lane_ray.py

This means that you have Flow properly configured with both SUMO and
rllib. Congratulations, you now have Flow set up!


Getting started (rllib version)
=================

See [getting started with rllib](http://ray.readthedocs.io/en/latest/rllib
.html#getting-started) for sample commands.

To visualize the training progress:
::

    tensorboard --logdir=~/ray_results

For information on how to deploy a cluster, see [instructions]
(http://ray.readthedocs.io/en/latest/autoscaling.html). The basic workflow is
 running the following locally, ssh-ing into the host machine, and starting
 jobs from there.
::

    ray create_or_update flow_ray_autoscale.yaml
    ray teardown flow_ray_autoscale.yaml


Installation (rllab version)
=================

Install rllab-multiagent
::

    git clone https://github.com/cathywu/rllab-multiagent.git
    cd rllab-multiagent

Create a conda environment named `flow`:
::

    conda env create -f environment.yml

For OSX
::

    brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi

For Linux
::

    sudo apt-get install swig
    sudo apt-get build-dep python-pygame
    sudo apt-get build-dep python-scipy

::

Now for both Linux and OSX, run
::
    python setup.py develop

Install flow within the rllab-multiagent repo
::

    git clone https://github.com/cathywu/flow.git  # Needs to be here for AWS experiments using rllab
    cd flow
    bash scripts/setup_sumo_osx.sh <DESIRED_PATH_TO_SUMO> # installs sumo at <DESIRED_PATH_TO_SUMO>/sumo
    python setup.py develop  # (install flow and dependencies)
    cp flow/core/config.template.py flow/core/config.py  # Create template for users using pycharm

Finally, add `<DESIRED_PATH_TO_SUMO>/sumo/tools` to your `PYTHONPATH` to give
Python access to TraCI and sumolib.
