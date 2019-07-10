#!/bin/bash
echo "Installing system dependencies for SUMO"
# rllab dependencies
brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi
# sumo dependencies
brew install Caskroom/cask/xquartz autoconf automake pkg-config libtool gdal proj xerces-c fox

echo "Installing sumo binaries and python tools"
mkdir -p $HOME/sumo_binaries
pushd $HOME/sumo_binaries
wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.4.1/binaries-mac.tar.xz
tar -xf binaries-mac.tar.xz
rm binaries-mac.tar.xz
chmod +x bin/*
popd

export SUMO_HOME="$HOME/sumo_binaries"
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$PYTHONPATH:$HOME/sumo_binaries/tools"

echo 'Add the following to your ~/.bashrc:'
echo ''
echo 'export SUMO_HOME="$HOME/sumo_binaries"'
echo 'export PATH="$SUMO_HOME/bin:$PATH"'
echo 'export PYTHONPATH="$PYTHONPATH:$HOME/sumo_binaries/tools"'
