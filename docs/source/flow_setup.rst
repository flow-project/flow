..    include:: <isonum.txt>
.. contents:: Table of contents

Local Installation of Flow
==========================

To get Flow running, you need three things: Flow,

SUMO, and (recommended to explore the full suite of Flow's capabilities)
a reinforcement learning library (RLlib). If you wish to use Flow with
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
    conda activate flow
    python setup.py develop

If the conda install fails, you can also install the requirements using pip by calling

::

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
If you are a Mac user and the above command gives you the error
``FXApp:openDisplay: unable to open display :0.0``, make sure to open the
application XQuartz.

*Troubleshooting*:
If you are a Mac user and the above command gives you the error
``Segmentation fault: 11``, make sure to reinstall ``fox`` using brew.
::

  # Uninstall Catalina bottle of fox:
  $ brew uninstall --ignore-dependencies fox

  # Edit brew Formula of fox:
  $ brew edit fox

  # Comment out or delete the following line: sha256 "c6697be294c9a0458580564d59f8db32791beb5e67a05a6246e0b969ffc068bc" => :catalina
  # Install Mojave bottle of fox:
  $ brew install fox


Testing your SUMO and Flow installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the above modules have been successfully installed, we can test the
installation by running a few examples. Before trying to run any examples, be
sure to enter your conda environment by typing:

::

    conda activate flow

Let’s see some traffic action:

::

    python examples/simulate.py ring

Running the following should result in the loading of the SUMO GUI.
Click the run button and you should see unstable traffic form after a
few seconds, a la (Sugiyama et al, 2008). This means that you have Flow
properly configured with SUMO!


(Optional) Installing Aimsun
----------------------------

In addition to SUMO, Flow supports the use of the traffic simulator "Aimsun".
In order setup Flow with Aimsun, you will first need to install Aimsun. This
can be achieved by following the installation instructions located in:
https://www.aimsun.com/aimsun-next/download/.

Once Aimsun has been installed, copy the path to the `Aimsun_Next` main
directory and place it in under the `AIMSUN_NEXT_PATH` variable in your bashrc.
This will allow Flow to locate and use this binary
during the execution of various tasks. The path should look something like:

::

    export AIMSUN_NEXT_PATH="/home/user/Aimsun_Next_X_Y_Z/"                   # Linux
    export AIMSUN_NEXT_PATH="/Applications/Aimsun Next.app/Contents/MacOS/"   # OS X

`Note for Mac users:` when you download Aimsun, you will get a folder named
"Programming". You need to rename it to "programming" (all lowercase) and to
move it inside the "Aimsun Next.app/Contents/MacOS/" directory so that the
python API can work.

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

Important note: For running Aimsun experiments, the `flow` environment should be
used, NOT the `aimsun_flow` environment that was just created.
The latter command should return an output similar to:

::

    /path/to/envs/aimsun_flow/bin/python

Copy the path up until right before /lib (i.e. /path/to/envs/aimsun_flow) and
place it under the `AIMSUN_SITEPACKAGES` variable in your bashrc, like this:

::

    export AIMSUN_SITEPACKAGES="/path/to/envs/aimsun_flow"

Testing your Aimsun installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test that you installation was successful, you can try running one of the
Aimsun examples within the Flow main directory. In order to do so, you need
to activate the `flow` env. Type:

::

    source deactivate aimsun_flow
    source activate flow
    python examples/simulate.py ring --aimsun

*Troubleshootig for Ubuntu users with Aimsun 8.4*: when you run the above
example, you may get a subprocess.Popen error ``OSError: [Errno 8] Exec format error:``.
To fix this, go to the `Aimsun Next` main directory, open the `Aimsun_Next`
binary with a text editor and add the shebang to the first line of the script
``#!/bin/sh``.

(Optional) Install Ray RLlib
----------------------------

Flow has been tested on a variety of RL libraries, the installation of which is
optional but may be of use when trying to execute some of the examples files
located in Flow.
RLlib is one such library.
First visit <https://github.com/flow-project/ray/blob/master/doc/source/installation.rst> and
install the required packages.

If you are not intending to develop RL algorithms or customize rllib you don't
need to do anything, Ray was installed when you created the conda environment.

If you are intending to modify Ray, the installation process for this library
is as follows:

::

    cd ~
    git clone https://github.com/flow-project/ray.git
    cd ray/python/
    python setup.py develop

If missing libraries cause errors, please also install additional
required libraries as specified at
<http://ray.readthedocs.io/en/latest/installation.html> and
then follow the setup instructions.

Testing your RLlib installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `getting started with RLlib <http://ray.readthedocs.io/en/latest/rllib.html#getting-started>`_ for sample commands.

To run any of the RL examples, make sure to run

::

    conda activate flow

In order to test run an Flow experiment in RLlib, try the following command:

::

    python examples/train.py singleagent_ring


If it does not fail, this means that you have Flow properly configured with
RLlib.


Visualizing with Tensorboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To visualize the training progress:

::

    tensorboard --logdir=~/ray_results

If tensorboard is not installed, you can install with pip:

::

    pip install tensorboard

For information on how to deploy a cluster, refer to the
`Ray instructions <http://ray.readthedocs.io/en/latest/autoscaling.html>`_.
The basic workflow is running the following locally, ssh-ing into the host
machine, and starting jobs from there.

::

    pip install boto3
    ray create-or-update scripts/ray_autoscale.yaml
    ray teardown scripts/ray_autoscale.yaml


(Optional) Install Stable Baselines
-----------------------------------

An additional library that Flow supports is the fork of OpenAI's Baselines, Stable-Baselines.
First visit <https://stable-baselines.readthedocs.io/en/master/guide/install.html> and
install the required packages and pip install the stable baselines package as described in their
installation instructions.

Testing your Stable Baselines installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can test your installation by running

::

    python examples/train.py singleagent_ring --rl_trainer Stable-Baselines


(Optional) Install h-baselines
------------------------------

h-baselines is another variant of stable-baselines that support the use of
single-agent, multiagent, and hierarchical policies. To install h-baselines,
run the following commands:

::

    git clone https://github.com/AboudyKreidieh/h-baselines.git
    cd h-baselines
    source activate flow  # if using a Flow environment
    pip install -e .


Testing your h-baselines installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can test your installation by running

::

    python examples/train.py singleagent_ring --rl_trainer h-baselines


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

Alternatively, the following segment of installation instructions is
also compatible with OSX installation, following the brew updates and
installations shown below.
<https://sumo.dlr.de/wiki/Installing/Linux_Build#Building_the_SUMO_binaries_with_cmake_.28recommended.29>

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
======================================================

To install a containerized Flow stack, run:

::

    docker run -d -p 5901:5901 -p 6901:6901 fywu85/flow-desktop:latest

To access the docker container, go to the following URL and enter the default password `password`:

::

    http://localhost:6901/vnc.html

To use the Jupyter Notebook inside the container, run:

::

    jupyter notebook --ip=127.0.0.1
