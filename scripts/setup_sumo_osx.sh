#!/bin/bash
echo "Installing system dependencies for SUMO"
brew install Caskroom/cask/xquartz autoconf automake pkg-config libtool gdal proj xerces-c fox

echo "Installing sumo binaries"
mkdir bin
pushd bin
wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.2.0/binaries-mac.tar.xz
tar -xf binaries-ubuntu1404.tar.xz
rm binaries-ubuntu1404.tar.xz
chmod +x *
export PATH="$PATH:$(pwd)"
export SUMO_HOME="$(pwd)"
popd
echo 'PYTHONPATH=$PYTHONPATH:'$(pwd)'/bin' >> ~/.bashrc
echo 'SUMO_HOME='$(pwd)'/bin' >> ~/.bashrc
