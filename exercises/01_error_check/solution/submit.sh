#!/bin/bash

#SBATCH -J 01_error_check
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2104x

srun -N1 -n1 ./vector_addition
