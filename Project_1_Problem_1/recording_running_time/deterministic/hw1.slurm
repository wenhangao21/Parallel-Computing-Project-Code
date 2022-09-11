#!/bin/bash
#
#SBATCH --job-name=wenhan_hw1
#SBATCH --output=wenhan_project_1_output.txt
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1
#SBATCH --time=00:30
#SBATCH -p short-24core


module load shared
# load a gcc module
module load mvapich2/gcc12.1/2.3.7

#set env variables which may help performance
export MV2_HOMOGENEOUS_CLUSTER=1
export MV2_ENABLE_AFFINITY=0

# compile the code
mpicc mpi_hello.c -o mpi_hello

mpirun ./mpi_hello
