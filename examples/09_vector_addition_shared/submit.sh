#!/bin/bash

#SBATCH -J 09_vector_addition_shared
#SBATCH -N 1
#SBATCH -t 5

srun -N1 -n1 ./vector_addition
