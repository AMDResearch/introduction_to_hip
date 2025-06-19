#!/bin/bash

#SBATCH -J 13_MM_kernels_streams
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi100
#SBATCH -p mi2101x
#SBATCH --reservation=cybercolombia

make

srun -N1 -n1 rocprof --stats --hip-trace --hsa-trace --sys-trace ./matrix_multiply
