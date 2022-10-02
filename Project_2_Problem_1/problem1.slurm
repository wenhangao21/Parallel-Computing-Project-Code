#!/bin/bash
#
#SBATCH --job-name=wenhan_project2
#SBATCH --output=p1_output.txt
#SBATCH --ntasks-per-node=37
#SBATCH --nodes=1
#SBATCH --time=00:30
#SBATCH -p long-96core

# request more time for large scale

module load shared
# load a gcc module
module load mvapich2/gcc12.1/2.3.7

#set env variables which may help performance
export MV2_HOMOGENEOUS_CLUSTER=1
export MV2_ENABLE_AFFINITY=0

# compile the code
mpicc p1_main.c -o p1_main

mpirun ./p1_main
