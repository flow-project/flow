#!/bin/bash
echo "Installing system dependencies for SUMO"
sudo apt-get update
sudo apt-get install -y cmake swig libgtest-dev python-pygame python-scipy
sudo apt-get install -y autoconf libtool pkg-config libgdal-dev libxerces-c-dev
sudo apt-get install -y libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev
sudo apt-get install -y build-essential curl unzip flex bison python python-dev
sudo apt-get install -y python3-dev
sudo pip3 install cmake cython

echo "Installing sumo binaries and python tools"
mkdir -p $HOME/sumo_binaries
pushd $HOME/sumo_binaries
wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.4.1/binaries-ubuntu1604.tar.xz
tar -xf binaries-ubuntu1604.tar.xz
rm binaries-ubuntu1604.tar.xz
chmod +x bin/*
python tools/build/setup-traci.py
popd

echo 'export PATH="$PATH:$HOME/sumo_binaries/bin"' >> ~/.bashrc
echo 'export SUMO_HOME="$HOME/sumo_binaries"' >> ~/.bashrc
