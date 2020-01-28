#!/bin/bash

# allows you to access python 2.7.4 which is needed for creating the aimsun_flow environment
conda config --append channels https://repo.anaconda.com/pkgs/free
conda config --append channels https://repo.anaconda.com/pkgs/pro

# create the conda environment
conda create -y -n aimsun_flow python=2.7.4

# install numpy within the environment
source activate aimsun_flow
pip install numpy
