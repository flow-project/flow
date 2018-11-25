cd ~
rm -rf ray
git clone https://github.com/eugenevinitsky/ray.git
cd ray/python
python setup.py develop
cd ~
sudo apt-get install -y \
    xorg-dev \
    libglu1-mesa \
    libgl1-mesa-dev \
    xvfb \
    libxinerama1 \
    libxcursor1 \
    gfortran
pip install imutils

# Running script in the background.
#nohup xvfb-run -a -s \
#    "-screen 0 1400x900x24 +extension RANDR"\
#    -- python script.py 0e-1 > script.log 2>&1 &
