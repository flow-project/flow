# Creating SUMO Wheels

This documentation walks you through creating SUMO binary folders and wheels 
for both Ubuntu and OSX. This procedure only needs to be performed whenever the
repository is migrating to a new SUMO version.

1. Clone the SUMO github repository:
        
        git clone https://github.com/eclipse/sumo.git

2. Checkout the version of SUMO you with to create wheels and binaries of:

        cd sumo
        git checkout <VERSION_NUMBER>

3. Install SUMO. If you are using Ubuntu, run the following commands:

        # install dependencies
        sudo apt-get update
        sudo apt-get install cmake swig libgtest-dev python-pygame python-scipy
        sudo apt-get install autoconf libtool pkg-config libgdal-dev libxerces-c-dev
        sudo apt-get install libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev
        sudo apt-get install build-essential curl unzip flex bison python python-dev
        sudo apt-get install python3-dev python3-pip

        # install sumo
        make -f Makefile.cvs
        ./configure
        make -j$nproc
        echo 'export SUMO_HOME="/path/to/sumo"' >> ~/.bashrc
        echo 'export PATH="/path/to/sumo/bin:$PATH"' >> ~/.bashrc
        echo 'export PYTHONPATH="/path/to/sumo/tools:$PYTHONPATH"' >> ~/.bashrc
        source ~/.bashrc

   Alternatively, if you are running Mac OSX, run the following commands:
   
        # install dependencies
        brew update
        brew install Caskroom/cask/xquartz autoconf automake pkg-config libtool
        brew install gdal proj xerces-c fox

        # install sumo
        make -f Makefile.cvs
        export CPPFLAGS=-I/opt/X11/include
        export LDFLAGS=-L/opt/X11/lib
        ./configure CXX=clang++ CXXFLAGS="-stdlib=libc++ -std=gnu++11" --with-xerces=/usr/local --with-proj-gdal=/usr/local
        make -j$nproc
        echo 'export SUMO_HOME="/path/to/sumo"' >> ~/.bash_profile
        echo 'export PATH="/path/to/sumo/bin:$PATH"' >> ~/.bash_profile
        echo 'export PYTHONPATH="/path/to/sumo/tools:$PYTHONPATH"' >> ~/.bash_profile
        source ~/.bash_profile

   Finally, verify that SUMO was successfully installed by running the 
   following command:
        
        sumo

4. Compress the SUMO binaries, data folder, and Python tools in a .tar.xz file.
   These are the files users will need to run SUMO, with the next step allowing 
   us to access TraCI as well.

        cd /path/to/sumo
        mkdir sumo_binaries 
        cp -r bin sumo_binaries && cp -r data sumo_binaries && cp -r tools sumo_binaries
        cd sumo_binaries
        tar -cJf binaries-<dist>.tar.xz !(Makefile*|start-command-line.bat)

5. Finally, the .tar.xf file needs to be placed on AWS so that other 
   individuals can have access to them. For now, send these files you created 
   in step 4 to Aboudy and he will place them and update the setup 
   commands accordingly, but we will need a more systematic way of doing this 
   in the future.

-------

**Additional Remarks**

- The binaries need to be created separately on Ubuntu 16.04 and 18.04 
  (they do not work on different distributions).
