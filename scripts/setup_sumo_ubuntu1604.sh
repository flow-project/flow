#!/bin/bash
echo "Installing system dependencies for SUMO"
sudo apt-get update
sudo apt-get install -y subversion autoconf build-essential libtool
sudo apt-get install -y libtool-bin libxerces-c3.1 libxerces-c3-dev
sudo apt-get install -y libproj-dev proj-bin proj-data libgdal1-dev
sudo apt-get install -y libfox-1.6-0 libfox-1.6-dev

echo "Installing sumo binaries"
mkdir bin
pushd bin
wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.2.0/binaries-ubuntu1604.tar.xz
tar -xf binaries-ubuntu1404.tar.xz
rm binaries-ubuntu1404.tar.xz
chmod +x *
popd
echo 'export PYTHONPATH=$PYTHONPATH:'$(pwd)'/bin' >> ~/.bashrc
echo 'export SUMO_HOME='$(pwd)'/bin' >> ~/.bashrc
