#!/bin/bash

# create the conda environment
conda create -n aimsun_flow python=2.7.4

# install numpy within the environment
source activate aimsun_flow
pip install numpy
