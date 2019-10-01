#!/bin/bash

# If the following error is encountered upon installation:
# 'PackagesNotFoundError: The following packages are not available from current channels: - python=2.7.4'
# Run the two lines of code below:

# conda config --append channels https://repo.anaconda.com/pkgs/free
# conda config --append channels https://repo.anaconda.com/pkgs/pro

# create the conda environment
conda create -y -n aimsun_flow python=2.7.4

# install numpy within the environment
source activate aimsun_flow
pip install numpy
