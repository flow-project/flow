#!/bin/bash
echo "Installing system dependencies for SUMO"
# rllab dependencies
brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi
# sumo dependencies
brew install Caskroom/cask/xquartz autoconf automake pkg-config libtool gdal proj xerces-c fox

echo "Installing sumo binaries"
mkdir -p $HOME/sumo_binaries/bin
pushd $HOME/sumo_binaries/bin
wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.3.0/binaries-mac.tar.xz
tar -xf binaries-mac.tar.xz
rm binaries-mac.tar.xz
chmod +x *
popd
echo 'export PATH=$PATH:$HOME/sumo_binaries/bin' >> ~/.bash_profile
echo 'export SUMO_HOME=$HOME/sumo_binaries/bin' >> ~/.bash_profile
