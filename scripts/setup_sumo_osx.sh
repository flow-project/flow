#!/bin/bash
echo "Installing system dependencies for SUMO"
# rllab dependencies
brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi
# sumo dependencies
brew install Caskroom/cask/xquartz autoconf automake pkg-config libtool gdal proj xerces-c fox

echo "Installing sumo binaries"
mkdir bin
pushd bin
wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.2.0/binaries-mac.tar.xz
tar -xf binaries-ubuntu1404.tar.xz
rm binaries-ubuntu1404.tar.xz
chmod +x *
popd
echo 'export PYTHONPATH=$PYTHONPATH:'$(pwd)'/bin' >> ~/.bashrc
echo 'export SUMO_HOME='$(pwd)'/bin' >> ~/.bashrc
