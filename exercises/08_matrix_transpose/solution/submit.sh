#!/bin/bash

#SBATCH -J 08_matrix_transpose
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2101x

srun -N1 -n1 ./matrix_transpose
