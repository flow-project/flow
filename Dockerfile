FROM continuumio/miniconda3:latest
MAINTAINER Fangyu Wu (fangyuwu@berkeley.edu)

# Binder
RUN pip install -U pip \
                   jupyter \
                   notebook
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}

# System
RUN apt-get update && \
	apt-get -y upgrade && \
	apt-get install -y \
    vim \
    gfortran \
    apt-utils


# Flow dependencies
RUN cd ~ && \
    conda install opencv && \
    pip install tensorflow

# Flow
RUN cd ~ && \
	git clone https://github.com/flow-project/flow.git && \
    cd flow && \
    git checkout binder && \
	pip install -e . --verbose

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
	git checkout cbe5b73 && \
    mkdir build/cmake-build && \
	cd build/cmake-build && \
	cmake ../.. && \
	make

# Ray/RLlib
RUN cd ~ && \
	pip install ray==0.6.2 \
                psutil
    
# Startup process
RUN	echo 'export SUMO_HOME="$HOME/sumo"' >> ~/.bashrc && \
	echo 'export PATH="$HOME/sumo/bin:$PATH"' >> ~/.bashrc && \
	echo 'export PYTHONPATH="$HOME/sumo/tools:$PYTHONPATH"' >> ~/.bashrc
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

