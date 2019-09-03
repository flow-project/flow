#!/bin/sh

#SBATCH --exclusive
#SBATCH -o out-%j
#SBATCH --constraint=opteron

source /etc/profile
module load anaconda3-2019a
source activate flow

export PATH=/home/gridsan/weizili/sumo-1_1_0/bin/:$PATH
export SUMO_HOME=/home/gridsan/weizili/sumo-1_1_0/ 
export PYTHONPATH=/home/gridsan/weizili/flow/:/home/gridsan/weizili/sumo-1_1_0/tools/
export LD_LIBRARY_PATH=/home/gridsan/weizili/software/proj-6.1.1/build/lib/

python ../examples/rllib/stabilizing_the_ring.py






