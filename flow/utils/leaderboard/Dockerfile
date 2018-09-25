FROM continuumio/miniconda3:4.5.4
MAINTAINER Fangyu Wu (fangyuwu@berkeley.edu)

# System
RUN apt-get update && \
	apt-get -y upgrade && \
	apt-get install -y \
    vim \
    apt-utils

# Flow
RUN cd ~ && \
	git clone https://github.com/flow-project/flow.git && \
    cd flow && \
	python setup.py develop

# SUMO dependencies
RUN apt-get install -y \
	cmake \
	build-essential \
	swig \
	libgdal-dev \
	libxerces-c-dev \
	libproj-dev \
	libfox-1.6-dev \
	libxml2-dev \
	libxslt1-dev \
	openjdk-8-jdk

# SUMO
RUN cd ~ && \
	git clone --recursive https://github.com/eclipse/sumo.git && \
	cd sumo && \
	git checkout 016c09d306 && \
    mkdir build/cmake-build && \
	cd build/cmake-build && \
	cmake ../.. && \
	make

# Ray/RLlib dependencies
RUN apt-get install -y \
	pkg-config \
	autoconf \
	curl \
	libtool \
	unzip \
	flex \
	bison \
	psmisc \
	python && \
	conda install -y \
	libgcc \
	cython

# Ray/RLlib
RUN cd ~ && \
	git clone https://github.com/eugenevinitsky/ray.git && \
	cd ray/python && \
	git checkout 6e07ea2 && \
	python setup.py develop
    
 # Startup process
RUN	echo 'export SUMO_HOME="$HOME/sumo"' >> ~/.bashrc && \
	echo 'export PATH="$HOME/sumo/bin:$PATH"' >> ~/.bashrc && \
	echo 'export PYTHONPATH="$HOME/sumo/tools:$PYTHONPATH"' >> ~/.bashrc && \
	echo 'export PYTHONPATH="/data:$PYTHONPATH"' >> ~/.bashrc && \
    echo '. ~/.bashrc' >> /startup.sh && \
	echo 'cd ~/flow/flow/utils/leaderboard' >> /startup.sh && \
	echo 'python run.py' >> /startup.sh && \
	chmod +x /startup.sh && \
    # Temporary solution to fix gym version
    pip install --upgrade gym

# Default command
CMD ["/startup.sh"]
