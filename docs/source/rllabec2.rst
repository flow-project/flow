Running rllab experiments on EC2
================================

This page covers the process of running rllab experients on an Amazon
EC2 instance. This tutorial assumes rllab has been installed correctly
(`instructions <https://rllab.readthedocs.io/en/latest/user/installation.html>`_).

Setting up rllab with AWS
-------------------------

First, follow the `rllab cluster setup
instructions <https://rllab.readthedocs.io/en/latest/user/cluster.html>`__
with region ``us-west-1``. Modify ``rllab/config_personal.py`` to
reference the most current Flow Docker image.

Navigate to your ``flow`` directory and modify ``Makefile.template`` per
the instructions in that file. The variable ``RLLABDIR`` should be the
relative path from your ``flow`` directory to ``rllab`` and **should not
have a backslash at the end**.

Running an experiment
---------------------

When running AWS experiments, your entire ``rllab`` directory is
uploaded to AWS so that the files necessary for your experiment are
available to the EC2 instance. Thus, commands are included to copy over
your ``flow`` directory to your ``rllab`` root directory (this is the
reason for the ``RLLABDIR`` variable above).

-  Before running an experiment, run ``make prepare`` from your ``flow``
   directory.
-  Ensure you have committed or otherwise tracked the state of your
   ``flow`` directory, because that instance is what will be used to run
   your experiment. Upon visualization, the same files will need to be
   used—for example, changes to your environment's state-space would break
   the ability to run a trained policy using a different state space.
   Check out an old commit of your ``flow`` directory before visualizing
   your experiment results.

``make clean`` removes the debug directory and also all XML files in
rllab root directory to reduce the size of the data to upload to AWS. If
you are using already-existing network files (from, say, OpenStreetMap),
ensure they do not get deleted by ``make clean`` by storing such files
elsewhere.

Inside the experiment, change the ``mode`` to ``ec2``. Other mode options are
``local``, which uses your standard environment and ``local_docker``, which 
uses a local Docker image to run the experiment. You should run the experiment in
``local_docker`` mode briefly before running the ``ec2`` version to
ensure there are no errors, particularly with Docker image compatibility.

After running ``python example.py`` once the ``mode`` of ``example.py``
is ``ec2``, you should see your experiment running on AWS.

Fetching Results
----------------

-  To get the results of your AWS experiments, navigate to your
   ``rllab`` directory and run ``python scripts/sync_s3.py``.
-  Your experiment results will be in ``data/s3`` in your ``rllab``
   directory.
