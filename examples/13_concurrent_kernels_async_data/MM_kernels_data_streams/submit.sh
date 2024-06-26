#!/bin/bash

#SBATCH -J 03_complete_matrix_multiply
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi100

make

srun -N1 -n1 rocprof --stats --hip-trace --hsa-trace --sys-trace ./matrix_multiply
