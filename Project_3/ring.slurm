#!/bin/bash
#
#SBATCH --job-name=ring
#SBATCH --output=ring.txt
#SBATCH --ntasks-per-node=64
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH -p long-96core

# request more time for large scale

module load shared
# load a gcc module
module load mvapich2/gcc12.1/2.3.7

#set env variables which may help performance
export MV2_HOMOGENEOUS_CLUSTER=1
export MV2_ENABLE_AFFINITY=0

# compile the code
mpicc ring.c -o ring -lm

mpirun ./ring
