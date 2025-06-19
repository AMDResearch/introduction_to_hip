#!/bin/bash

#SBATCH -J 01_vector_addition
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2101x

srun -N1 -n1 ./vector_addition
