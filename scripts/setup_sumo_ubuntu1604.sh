#!/bin/bash
echo "Installing system dependencies for SUMO"
sudo apt-get update
sudo apt-get install cmake python g++ libxerces-c-dev libfox-1.6-dev
sudo apt-get install libgdal-dev libproj-dev libgl2ps-dev swig

echo "Installing sumo binaries and python tools"
mkdir -p $HOME/sumo_binaries
pushd $HOME/sumo_binaries
git clone https://github.com/eclipse/sumo.git
cd sumo
git checkout 2147d155b1
cmake .
make -j$(nproc)
popd

echo 'export PATH="$PATH:$HOME/sumo_binaries/sumo/bin"' >> ~/.bashrc
echo 'export SUMO_HOME="$HOME/sumo_binaries/sumo"' >> ~/.bashrc
echo 'export PYTHONPATH="$PYTHONPATH:$HOME/sumo_binaries/sumo/tools"' >> ~/.bashrc
