#!/bin/bash

#SBATCH -J 03_complete_matrix_multiply
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2104x

srun -N1 -n1 ./matrix_multiply
