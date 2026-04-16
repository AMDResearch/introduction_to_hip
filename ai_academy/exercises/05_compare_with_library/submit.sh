#!/bin/bash

#SBATCH -J 05_compare_with_library
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2101x
#SBATCH --reservation=cybercolombia

srun -N1 -n1 rocprof --stats --hip-trace ./matrix_multiply
