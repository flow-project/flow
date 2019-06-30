..    include:: <isonum.txt>
.. contents:: Table of contents

Local Installation of Flow
==================

To get Flow running, you need three things: Flow,

SUMO, and (recommended to explore the full suite of Flow's capabilities)
a reinforcement learning library (RLlib/rllab). If you wish to use Flow with
the traffic simulator Aimsun, this can be achieved by following the setup
instructions under the "Installing Aimsun" subsection.
If you choose not to install a reinforcement learning library, you will
still be able to build and run SUMO-only traffic tasks, but will not be
able to run experiments which require learning agents. Once
each component is installed successfully, you might get some missing
module bugs from Python. Just install the missing module using
your OS-specific package manager / installation tool. Follow the
shell commands below to get started.

**It is highly recommended that users install**
`Anaconda <https://www.anaconda.com/download>`_ **or**
`Miniconda <https://conda.io/miniconda.html>`_
**for Python and the setup instructions will assume that you are
doing so.**

Installing Flow and SUMO
------------------------

In this section we install Flow as well as the binaries and packages needed
to support the traffic simulator used in modeling the dynamics of traffic
networks: SUMO.

If you have not done so already, download the Flow github repository.

::

    git clone https://github.com/flow-project/flow.git
    cd flow

We begin by creating a conda environment and installing Flow and its
dependencies within the environment. This can be done by running the below
script. Be sure to run the below commands from ``/path/to/flow``.

::

    # create a conda environment
    conda env create -f environment.yml
    source activate flow
    # install flow within the environment
    pip install -e .

Next, we install the necessary pre-compiled SUMO binaries and python tools. In order to
install everything you will need from SUMO, run one of the below scripts from
the Flow main directory. Choose the script that matches the operating system
you are running.

For Ubuntu 14.04:

::

    scripts/setup_sumo_ubuntu1404.sh

For Ubuntu 16.04:

::

    scripts/setup_sumo_ubuntu1604.sh

For Ubuntu 18.04:

::

    scripts/setup_sumo_ubuntu1804.sh

For Mac:

::

    scripts/setup_sumo_osx.sh

If you are using an unsupported operating system (e.g. Arch Linux), or the
binaries provided by the above scripts are not compatible with your machine, you
will have to personally build the SUMO binary files. For more, please see
`(Optional) Direct install of SUMO from GitHub`_ or refer to `SUMO's
documentation <http://sumo.dlr.de/wiki/Installing/Linux_Build>`_.

**WARNING**:
Flow is not currently compatible with the most up-to-date version of SUMO.

Finally, test your SUMO install and version by running the following commands.

::

    which sumo
    sumo --version
    sumo-gui


*Troubleshooting*:
Note that, if the above commands did not work, you may need to run
``source ~/.bashrc``  or open a new terminal to update your $PATH variable.

*Troubleshooting*:
If you are a Mac user and the above command gives you the error ``FXApp:openDisplay: unable to open display :0.0``, make sure to open the application XQuartz.

Testing your installation
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the above modules have been successfully installed, we can test the
installation by running a few examples. Before trying to run any examples, be
sure to enter your conda environment by typing:

::

    source activate flow

Let’s see some traffic action:

::

    python examples/sumo/sugiyama.py

Running the following should result in the loading of the SUMO GUI.
Click the run button and you should see unstable traffic form after a
few seconds, a la (Sugiyama et al, 2008). This means that you have Flow
properly configured with SUMO and Flow!


(Optional) Installing Aimsun
-----------------

In addition to SUMO, Flow supports the use of the traffic simulator "Aimsun".
In order setup Flow with Aimsun, you will first need to install Aimsun. This
can be achieved by following the installation instructions located in:
https://www.aimsun.com/aimsun-next/download/.

Once Aimsun has been installed, copy the path to the `Aimsun_Next` main
directory and place it in under the `AIMSUN_NEXT_PATH` variable in the
"flow/config.py" folder. This will allow Flow to locate and use this binary
during the execution of various tasks. The path should look something like:

::

    /home/user/Aimsun_Next_X_Y_Z/                   # Linux
    /Applications/Aimsun Next.app/Contents/MacOS/   # OS X

`Note for Mac users:` when you download Aimsun, you will get a folder named "Programming". You need to rename it to "programming" (all lowercase) and to move it inside the "Aimsun Next.app/Contents/MacOS/" directory so that the python API can work.

In addition, being that Aimsun's python API is written to support Python 2.7.4,
we will need to create a Python 2.7.4 conda environment that Aimsun can refer
to when executing commands. In order to do so, run the following script from
the Flow main directory:

::

    scripts/setup_aimsun.sh

You can then verify that the above command has successfully installed the
required conda env by typing:

::

    source activate aimsun_flow
    which python

The latter command should return an output similar to:

::

    /path/to/envs/aimsun_flow/bin/python

Copy the path up until right before /bin (i.e. /path/to/envs/aimsun_flow) and
place it under the `AIMSUN_SITEPACKAGES` variable in flow/config.py.


Testing your installation
~~~~~~~~~~~~~~~~~~~~~~~~~

To test that you installation was successful, you can try running one of the
Aimsun examples within the Flow main directory. In order to do so, type:

::

    python examples/aimsun/sugiyama.py


(Optional) Install Ray RLlib
----------------------------

Flow has been tested on a variety of RL libraries, the installation of which is
optional but may be of use when trying to execute some of the examples files
located in Flow.
RLlib is one such library.
First visit <https://github.com/flow-project/ray/blob/master/doc/source/installation.rst> and
install the required packages.

If you are not intending to develop RL algorithms or customize rllib you don't need to do anything,
Ray was installed when you created the conda environment.

If you are intending to modify Ray, the installation process for this library is as follows:

::

    cd ~
    git clone https://github.com/flow-project/ray.git
    cd ray/python/
    python setup.py develop

If missing libraries cause errors, please also install additional
required libraries as specified at
<http://ray.readthedocs.io/en/latest/installation.html> and
then follow the setup instructions.


Testing your installation
~~~~~~~~~~~~~~~~~~~~~~~~~

See `getting started with RLlib <http://ray.readthedocs.io/en/latest/rllib.html#getting-started>`_ for sample commands.

To run any of the RL examples, make sure to run

::

    source activate flow

In order to test run an Flow experiment in RLlib, try the following command:

::

    python examples/rllib/stabilizing_the_ring.py

If it does not fail, this means that you have Flow properly configured with
RLlib.

To visualize the training progress:

::

    tensorboard --logdir=~/ray_results

If tensorboard is not installed, you can install with pip:

::

    pip install tensorboard

For information on how to deploy a cluster, refer to the `Ray instructions <http://ray.readthedocs.io/en/latest/autoscaling.html>`_.
The basic workflow is running the following locally, ssh-ing into the host machine, and starting
jobs from there.

::

    pip install boto3
    ray create-or-update scripts/ray_autoscale.yaml
    ray teardown scripts/ray_autoscale.yaml


(Optional) Install Rllab-multiagent
-----------------------------------

`rllab-multiagent` is another RL library that is compatible with Flow.
In order to install the `rllab-multiagent` library, follow the below instructions:

::

    cd ~
    git clone https://github.com/cathywu/rllab-multiagent.git
    cd rllab-multiagent
    python setup.py develop

For linux run

::

    echo 'export PYTHONPATH="$HOME/rllab-multiagent:$PYTHONPATH"' >> ~/.bashrc
    source ~/.bashrc

For mac run

::

    echo 'export PYTHONPATH="$HOME/rllab-multiagent:$PYTHONPATH"' >> ~/.bash_profile
    source ~/.bash_profile


Testing your installation
~~~~~~~~~~~~~~~~~~~~~~~~~

To run any of the RL examples, make sure to run

::

    source activate flow

In order to test run an Flow experiment in rllab-multiagent, try the following
command:

::

    python examples/rllab/stabilizing_the_ring.py

If it does not fail, this means that you have Flow properly configured with
rllab-multiagent.


(Optional) Direct install of SUMO from GitHub
---------------------------------------------

The below commands walk you through installing and building SUMO locally. Note
that if this does not work, you are recommended to point an issue on the
flow-dev message board or refer to SUMO's
`documentation <http://sumo.dlr.de/wiki/Installing/Linux_Build>`_ regarding
installing their software.

We begin by downloading SUMO's github directory:

::

    cd ~
    git clone https://github.com/eclipse/sumo.git
    cd sumo
    git checkout 2147d155b1
    make -f Makefile.cvs

If you have OSX, run the following commands. If you don't have brew
you can find installation instructions at
<https://docs.brew.sh/Installation>

::

    brew update
    brew install Caskroom/cask/xquartz
    brew install autoconf
    brew install automake
    brew install pkg-config
    brew install libtool
    brew install gdal
    brew install proj
    brew install xerces-c
    brew install fox
    export CPPFLAGS=-I/opt/X11/include
    export LDFLAGS=-L/opt/X11/lib
    ./configure CXX=clang++ CXXFLAGS="-stdlib=libc++ -std=gnu++11" --with-xerces=/usr/local --with-proj-gdal=/usr/local
    make -j$nproc
    echo 'export SUMO_HOME="$HOME/sumo"' >> ~/.bash_profile
    echo 'export PATH="$HOME/sumo/bin:$PATH"' >> ~/.bash_profile
    echo 'export PYTHONPATH="$HOME/sumo/tools:$PYTHONPATH"' >> ~/.bash_profile
    source ~/.bash_profile

If you have Ubuntu 14.04+, run the following command

::

    ./configure
    make -j$nproc
    echo 'export SUMO_HOME="$HOME/sumo"' >> ~/.bashrc
    echo 'export PATH="$HOME/sumo/bin:$PATH"' >> ~/.bashrc
    echo 'export PYTHONPATH="$HOME/sumo/tools:$PYTHONPATH"' >> ~/.bashrc
    source ~/.bashrc


Virtual installation of Flow (using docker containers)
================================

To install a containerized Flow stack, run:
::
    docker run -d -p 5901:5901 -p 6901:6901 fywu85/flow-desktop:latest

To access the docker container, go to the following URL and enter the default password `password`:
::
    http://localhost:6901/vnc.html

To use the Jupyter Notebook inside the container, run:
::
    jupyter notebook --ip=127.0.0.1
