Setup Instructions
******************

To get Flow running, you need three things: Flow,
SUMO, and (optionally) a reinforcement learning library (RLlib/rllab).
If you choose not to install a reinforcement learning library, you will 
still be able to build and run SUMO-only traffic tasks, but will not be
able to run experiments which require learning agents. Once
each component is installed successfully, you might get some missing
module bugs from Python. Just install the missing module using
your OS-specific package manager / installation tool. Follow the 
shell commands below to get started.

Dependencies
============
We begin by installing dependencies needed by the four repositories mentioned
above. It will be useful to install `Anaconda <https://www.anaconda.com/download>`_
for Python.

For Ubuntu 16.04:
::

    sudo apt-get update && sudo apt-get upgrade
    sudo apt-get install cmake swig libgtest-dev python-pygame python-scipy autoconf libtool pkg-config libgdal-dev libxerces-c-dev libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev build-essential curl unzip flex bison python python-dev python3-dev
    sudo pip3 install cmake cython

For OSX (feel free to ignore the rllab dependencies if you don't wish to
install it):
::

    # rllab dependencies
    brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi
    # sumo dependencies
    brew install Caskroom/cask/xquartz autoconf automake pkg-config libtool gdal proj xerces-c fox

sumo
====
Next, we install SUMO, an open source traffic microsimulator which will be used
the update the states of vehicles, traffic lights, and other RL and
human-driven agents during the simulation process.
::

    cd ~
    git clone https://github.com/eclipse/sumo.git
    cd sumo
    git checkout 1d4338ab80
    make -f Makefile.cvs

If you have OSX, run the following commands
::
    export CPPFLAGS=-I/opt/X11/include
    export LDFLAGS=-L/opt/X11/lib

Now for both OSX and linux run the following command
::
    ./configure
    make -j$nproc
    echo 'export SUMO_HOME="$HOME/sumo"' >> ~/.bashrc
    echo 'export PATH="$HOME/sumo/bin:$PATH"' >> ~/.bashrc
    echo 'export PYTHONPATH="$HOME/sumo/tools:$PYTHONPATH"' >> ~/.bashrc
    source ~/.bashrc

Test your sumo install and version by running the following commands
::
    which sumo
    sumo --version
    sumo-gui

Flow
====
Once sumo and the various dependencies are in place, we are ready to install a
functional version of Flow. With this, we can begin to simulate traffic within
sumo using OpenAI gym-compatible environments. Note that separate RL algorithms
will be needed to train autonomous agents within the simulation to improve
various traffic flow properties (see the sections on rllab-multiagent and
Ray/RLlib for more).
::

    cd ~
    git clone https://github.com/berkeleyflow/flow.git
    cd flow
    python3 setup.py develop
    echo 'export PYTHONPATH="$HOME/flow:$PYTHONPATH"' >> ~/.bashrc
    source ~/.bashrc


Testing the Installation
========================

Once the above modules have been successfully installed, we can test the
installation by running a few examples.

Let’s see some traffic action:
::

    python examples/sumo/sugiyama.py

Running the following should result in the loading of the SUMO GUI.
Click the run button and you should see unstable traffic form after a
few seconds, a la (Sugiyama et al, 2008). This means that you have Flow
properly configured with SUMO.

Optionally, run the unit tests:
::

    nose2 -s tests/fast_tests

Congratulations, you now have successfully set up Flow!


rllab-multiagent (optional)
===========================
Flow has been tested on a variety of RL libraries, the installation of which is
optional but may be of use when trying to execute some of the examples files
located in Flow. rllab-multiagent is one of these such libraries.  In order
to install the `rllab-multiagent` library, follow the below instructions
::

    cd ~
    git clone https://github.com/cathywu/rllab-multiagent.git
    cd rllab-multiagent
    conda env create -f environment.yml
    source activate flow-rllab
    python3 setup.py develop
    echo 'export PYTHONPATH="$HOME/rllab-multiagent:$PYTHONPATH"' >> ~/.bashrc
    source ~/.bashrc
    source activate flow-rllab

Ray/RLlib (optional)
====================
RLlib is another RL library that has been extensively tested on the Flow
repository. 
First visit <http://ray.readthedocs.io/en/latest/installation.html> and
install the required packages. 
The installation process for this library is as follows:
::

    cd ~
    git clone https://github.com/eugenevinitsky/ray.git
    pushd ray/python
    sudo python3 setup.py develop
    popd

If missing libraries cause errors, please also install additional 
required libraries as specified at 
<http://ray.readthedocs.io/en/latest/installation.html> and
then follow the setup instructions.

Getting started (rllab-multiagent)
==================================

To run any of the RL examples, make sure to run
::

    source activate flow
    
In order to test run an Flow experiment in rllab-multiagent, try the following
command:
::

    python examples/rllab/stabilizing_the_ring.py

If it does not fail, this means that you have Flow properly configured with
rllab-multiagent.


Getting started (Ray/RLlib)
===========================

See `getting started with RLlib <http://ray.readthedocs.io/en/latest/rllib.html#getting-started>`_ for sample commands.

In order to test run an Flow experiment in RLlib, try the following command:
::

    python examples/rllib/stabilizing_the_ring.py

If it does not fail, this means that you have Flow properly configured with
RLlib.

To visualize the training progress:
::

    tensorboard --logdir=~/ray_results

For information on how to deploy a cluster, refer to the `Ray instructions <http://ray.readthedocs.io/en/latest/autoscaling.html>`_.
The basic workflow is running the following locally, ssh-ing into the host machine, and starting
jobs from there.

::

    ray create_or_update scripts/ray_autoscale.yaml
    ray teardown scripts/ray_autoscale.yaml


Custom configuration
====================

You may define user-specific config parameters as follows
::

    cp flow/core/config.template.py flow/core/config.py  # Create template for users using pycharm
