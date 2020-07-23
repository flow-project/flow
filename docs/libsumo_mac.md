# How to install Libsumo for Mac OS

This is adapted from an email exchange with the SUMO staff.



To install libsumo requires re-building and installing SUMO from source.

## Steps
 
- **Install swig:** ```brew install swig```
- **Clone the repo:** ```git clone https://github.com/eclipse/sumo.git```
- **Create a “cmake-build” directory inside sumo/build/ and navigate to it:** ```mkdir build/cmake-build && cd build/cmake-build```

**The next 3 steps are inside that directory**

- ```cmake ../..```
- ```make```
- ```make install```

## Additional Notes
- You can test if libsumo has been built looking at (./testlibsumo) inside the sumo/bin/ directory.
- Bear in mind to use libsumo with the same Python version with which CMake built SUMO.