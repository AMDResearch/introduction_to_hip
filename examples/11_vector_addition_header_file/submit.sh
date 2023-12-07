#!/bin/bash

#SBATCH -J 11_vector_addition_header_file
#SBATCH -N 1
#SBATCH -t 5

srun -N1 -n1 ./vector_addition
