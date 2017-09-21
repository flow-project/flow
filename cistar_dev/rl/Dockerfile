FROM maven:3
# FROM ubuntu:16.04

# ========== Anaconda ==========
# https://github.com/ContinuumIO/docker-images/blob/master/anaconda/Dockerfile
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget  --no-check-certificate --quiet https://repo.continuum.io/archive/Anaconda2-2.5.0-Linux-x86_64.sh && \
    /bin/bash /Anaconda2-2.5.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm /Anaconda2-2.5.0-Linux-x86_64.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8
ENTRYPOINT [ "/usr/bin/tini", "--" ]

# ========== Upgrade pip ==========
RUN pip install --upgrade pip
RUN apt-get -y install git make cmake unzip apt-utils

# ========== SUMO ==========
ENV SUMO_VERSION 0.27.1
ENV SUMO_SRC sumo-src-$SUMO_VERSION
ENV SUMO_HOME /opt/sumo

# Install system dependencies.
RUN apt-get update && apt-get -y install -qq \
    g++ libxerces-c3.1 libxerces-c3-dev \
    libproj-dev proj-bin proj-data libtool libgdal1-dev \
    libfox-1.6-0 libfox-1.6-dev

# Download and extract source code
RUN wget http://downloads.sourceforge.net/project/sumo/sumo/version%20$SUMO_VERSION/sumo-src-$SUMO_VERSION.tar.gz
RUN tar xzf sumo-src-$SUMO_VERSION.tar.gz && \
    mv sumo-$SUMO_VERSION $SUMO_HOME && \
    rm sumo-src-$SUMO_VERSION.tar.gz

# Configure and build from source.
RUN cd $SUMO_HOME && ./configure && make install

# RUN echo "export PYTHONPATH=\"/opt/sumo/sumo-$SUMO_VERSION/tools\"" >> /root/.bashrc
ENV PYTHONPATH $SUMO_HOME/tools:$PYTHONPATH

# Ensure the installation works. If this call fails, the whole build will fail.
RUN sumo

# Add volume to allow for host data to be used
# RUN mkdir /data
# VOLUME /data

# Expose a port so that SUMO can be started with --remote-port 8873 to be controlled from outside Docker
EXPOSE 8873

# ========== Special Deps ==========
RUN pip install awscli
# ALE requires zlib
RUN apt-get -y install zlib1g-dev
RUN apt-get install -y vim ack-grep

# Commented out because we don't need MUJOCO / Pygame
# MUJOCO requires graphics stuff (Why?)
# RUN apt-get -y build-dep glfw
# RUN apt-get -y install libxrandr2 libxinerama-dev libxi6 libxcursor-dev
# copied from requirements.txt
#RUN pip install imageio tabulate nose
# usual pip install pygame will fail
#RUN conda install --yes -c https://conda.anaconda.org/kne pybox2d
#RUN conda install --yes -impc https://conda.binstar.org/tlatorre pygame
#RUN apt-get build-dep -y python-pygame
#RUN pip install Pillow

# ========== OpenAI Gym ==========
RUN pip install gym
RUN apt-get -y install ffmpeg
RUN apt-get -y install libav-tools

# Dependencies of OpenCV
RUN apt-get -y install libgtk2.0-0

# ========== Add codebase stub ==========
CMD mkdir /root/code
ADD environment.yml /root/code/environment.yml
RUN conda env create -f /root/code/environment.yml

ENV PYTHONPATH /root/code/rllab:$PYTHONPATH
ENV PATH /opt/conda/envs/rllab3/bin:$PATH
RUN echo "source activate rllab3" >> /root/.bashrc
ENV BASH_ENV /root/.bashrc
WORKDIR /root/code

RUN /opt/conda/envs/rllab3/bin/pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp35-cp35m-linux_x86_64.whl

RUN apt-get install -y libopenblas-dev
RUN printf "[blas]\nldflags = -lopenblas\n" > ~/.theanorc
