#!/bin/bash
echo "Installing system dependencies for SUMO"
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y subversion autoconf build-essential libtool libtool-bin
sudo apt-get install -y libxerces-c3.1 libxerces-c3-dev libproj-dev proj-bin proj-data libgdal1-dev libfox-1.6-0 libfox-1.6-dev
# clang

# TODO: try with pkg-config

BASH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "\nDownloading SUMO to $1. This may take some time."
echo "You may be prompted (twice) to authorize downloading from the repository (press (t) to temporarily accept)."
mkdir -p $1
echo "Temporarily changing directories"
pushd $1
svn checkout https://svn.code.sf.net/p/sumo/code/trunk/sumo@25706 > /dev/null
pushd sumo

echo "\nPatching SUMO for flow compatibility"
patch -p1 < $BASH_DIR/departure_time_issue.patch

make -f Makefile.cvs
./configure
make -j16

echo "\n#############################\n"
echo "add $1/sumo/tools to your PYTHONPATH and set SUMO_HOME to complete the installation!\n"

echo "This can be done by appending the following to your bash_profile:\n "
echo "export PYTHONPATH=$1/sumo/tools\n"
echo "export SUMO_HOME=\"$1/sumo\""
echo "#############################\n"
echo "Returning to flow directory"

popd
popd
