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
svn checkout https://svn.code.sf.net/p/sumo/code/trunk/sumo@26784 > /dev/null
pushd sumo

echo "Patching SUMO for flow compatibility"
patch -p1 < $BASH_DIR/departure_time_issue.patch


echo "Building SUMO"
export CPPFLAGS="$CPPFLAGS -I/opt/X11/include/"
export LDFLAGS="-L/opt/X11/lib"

autoreconf -i
./configure CXX=clang++ CXXFLAGS="-stdlib=libc++ -std=gnu++11" --with-xerces=/usr/local --with-proj-gdal=/usr/local

make -j`sysctl -n hw.ncpu`
make install

echo "\n#############################\n"
echo "add $1/sumo/tools to your PYTHONPATH and set SUMO_HOME to complete the installation!\n"

echo "This can be done by appending the following to your bash_profile:\n "
echo "export PYTHONPATH=$1/sumo/tools\n"
echo "export SUMO_HOME=\"$1/sumo\""
echo "#############################\n"
echo "Returning to flow directory"

popd
popd
