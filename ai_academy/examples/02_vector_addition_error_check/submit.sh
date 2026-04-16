#!/bin/bash

#SBATCH -J 02_vector_addition_error_check
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2101x
#SBATCH --reservation=cybercolombia

srun -N1 -n1 ./vector_addition
