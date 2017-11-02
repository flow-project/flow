#!/bin/bash
echo "Installing system dependencies for SUMO"
# Read in desired path
brew install Caskroom/cask/xquartz autoconf automake pkg-config libtool gdal proj xerces-c fox

BASH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Downloading SUMO to $1. This may take some time."
echo "You may be prompted (twice) to authorize downloading from the repository (press (t) to temporarily accept)."
mkdir -p $1
echo "Temporarily changing directories"
pushd $1
# TODO update with nightly build
svn checkout https://svn.code.sf.net/p/sumo/code/trunk/sumo@25706 > /dev/null
pushd sumo

echo "Patching SUMO for flow compatibility"
# TODO add the patch code
# TODO add patch for ../tools/build/version.py as well
patch -p1 < $BASH_DIR/departure_time_issue.patch


echo "Building SUMO"
export CPPFLAGS="$CPPFLAGS -I/opt/X11/include/"
export LDFLAGS="-L/opt/X11/lib"

autoreconf -i > /dev/null
./configure CXX=clang++ CXXFLAGS="-stdlib=libc++ -std=gnu++11" --with-xerces=/usr/local --with-proj-gdal=/usr/local > /dev/null

make -j`sysctl -n hw.ncpu` > /dev/null
make install > /dev/null

echo "\n#############################\n"
echo "add $1/sumo/tools to your PYTHON_PATH to complete the installation!\n"

echo "This can be done by appending the following to your bash_profile:\n "
echo "export PYTHON_PATH=$1/sumo/tools\n"
echo "#############################\n"
echo "Returning to flow directory"

popd
popd
