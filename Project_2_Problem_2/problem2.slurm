#!/bin/bash
#
#SBATCH --job-name=wenhan_project2
#SBATCH --output=p2_output.txt
#SBATCH --ntasks-per-node=28
#SBATCH --nodes=1
#SBATCH --time=00:15
#SBATCH -p short-28core

# request more time for large scale

module load shared
# load a gcc module
module load mvapich2/gcc12.1/2.3.7

#set env variables which may help performance
export MV2_HOMOGENEOUS_CLUSTER=1
export MV2_ENABLE_AFFINITY=0

# compile the code
mpicc p2_main.c -o p2_main

mpirun ./p2_main
