#!/bin/bash

echo "Installing system dependencies for SUMO"
sudo apt-get update
sudo apt-get install -y cmake swig libgtest-dev python-pygame python-scipy
sudo apt-get install -y autoconf libtool pkg-config libgdal-dev libxerces-c-dev
sudo apt-get install -y libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev
sudo apt-get install -y build-essential curl unzip flex bison python python-dev
sudo apt-get install -y python3-dev
sudo pip3 install cmake cython

echo "Installing sumo binaries"
rm -rf $HOME/sumo_binaries/bin
mkdir -p $HOME/sumo_binaries/bin
pushd $HOME/sumo_binaries/bin
wget https://flow-sumo.s3-us-west-1.amazonaws.com/libsumo_test/binaries-ubuntu1604.tar.gz
tar xvf binaries-ubuntu1604.tar.gz
rm binaries-ubuntu1604.tar.gz
popd
cd sumo_binaries
chmod +x *

echo '# Added by Sumo / Libsumo instalation' >> ~/.bashrc
echo 'export PATH="$HOME/sumo_binaries/bin:$PATH"' >> ~/.bashrc
echo 'export SUMO_HOME="$HOME/sumo_binaries/bin"' >> ~/.bashrc
echo 'export PYTHONPATH="$PYTHONPATH:$HOME/sumo_binaries/tools"' >> ~/.bashrc
