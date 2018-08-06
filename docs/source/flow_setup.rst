Setup Instructions
******************

To get Flow running, you need three things: Flow,
SUMO, and a reinforcement learning library (RLlib/rllab). Once each 
component is installed successfully, you might get some missing 
module bugs from Python. Just install the missing module using 
your OS-specific package manager / installation tool. Follow the 
shell commands below to get started.

Dependencies
============
We begin by installing dependencies needed by the four repositories mentioned
above.
It will be useful to install `Anaconda <https://www.anaconda.com/download>`_ for Python and enable it right away.
For Ubuntu 16.04:
::

    sudo apt-get update && sudo apt-get upgrade
    sudo apt-get install cmake swig libgtest-dev python-pygame python-scipy autoconf libtool pkg-config libgdal-dev libxerces-c-dev libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev build-essential curl unzip flex bison python python-dev python3-dev
    sudo pip3 install cmake cython

For OSX:
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
    git clone https://github.com/DLR-TS/sumo.git
    cd sumo 
    git checkout 1d4338ab80
    make -f Makefile.cvs
    export CPPFLAGS=-I/opt/X11/include
    export LDFLAGS=-L/opt/X11/lib
    ./configure
    make
    echo 'export SUMO_HOME="$HOME/sumo"' >> ~/.bashrc
    echo 'export PATH="$HOME/sumo/bin:$PATH"' >> ~/.bashrc
    echo 'export PYTHONPATH="$HOME/sumo/tools:$PYTHONPATH"' >> ~/.bashrc
    source ~/.bashrc

rllab-multiagent
================
Flow has been tested on a variety of RL libraries, the installation of which is
optional but may be of use when trying to execute some of the examples files
located in Flow. rllab-multiagent is one of these such libraries.  In order 
to install the `rllab-multiagent` library, follow the below instructions
::

    cd ~
    git clone https://github.com/cathywu/rllab-multiagent.git
    cd rllab-multiagent
    conda env create -f environment.yml
    python3 setup.py develop
    echo 'export PYTHONPATH="$HOME/rllab-multiagent:$PYTHONPATH"' >> ~/.bashrc
    source ~/.bashrc
    ```

    ## flow
    ```shell
    cd ~/rllab-multiagent # Repo flow needs to be installed within rllab-multiagent.
    git clone https://github.com/berkeleyflow/flow.git
    cd flow
    python3 setup.py develop
    echo 'export PYTHONPATH="$HOME/rllab-multiagent/flow:$PYTHONPATH"' >> ~/.bashrc
    source ~/.bashrc

Ray/RLlib
=========
RLlib is another RL library that has been extensively tested on the Flow
repository. The installation process for this library is as follows:
::

    cd ~
    git clone https://github.com/eugenevinitsky/ray.git
    pushd ray/python
    sudo python3 setup.py develop
    popd

If missing libraries cause errors, please also install the required libraries as specified at <http://ray.readthedocs.io/en/latest/installation.html> and then follow the setup instructions.

Testing the Installation
========================

Once the above modules have been successfully installed, we can test the
installation by running a few examples.

To run any of the RL examples, make sure to run
::

    source activate flow
    
Running the following should result in the loading of the SUMO GUI.
Click the run button and you should see unstable traffic form after a
few seconds, a la (Sugiyama et al, 2008).

Run the unit tests:
::

    nose2 -s tests/fast_tests

Let’s see some traffic action:
::

    python examples/sumo/sugiyama.py

This means that you have Flow properly configured with SUMO.
::

    python examples/rllib/stabilizing_the_ring.py

This means that you have Flow properly configured with both SUMO and
rllib. Congratulations, you now have Flow set up!


Getting started (Ray/RLlib)
===========================

See `getting started with RLlib <http://ray.readthedocs.io/en/latest/rllib.html#getting-started>`_ for sample commands.

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
