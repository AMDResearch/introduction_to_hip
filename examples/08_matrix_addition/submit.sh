#!/bin/bash

#SBATCH -J 08_matrix_addition
#SBATCH -N 1
#SBATCH -t 5
#SBATCH --reservation=sc23

srun -N1 -n1 ./matrix_addition
