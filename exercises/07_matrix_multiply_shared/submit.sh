#!/bin/bash

#SBATCH -J 07_matrix_multiply_shared
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2101x
#SBATCH --reservation=cybercolombia

srun -N1 -n1 ./matrix_multiply
