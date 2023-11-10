#!/bin/bash

#SBATCH -A staff
#SBATCH -J 01_vector_addition
#SBATCH -N 1
#SBATCH -t 5

srun -N1 -n1 ./vector_addition
