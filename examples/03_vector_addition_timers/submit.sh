#!/bin/bash

#SBATCH -J 03_vector_addition_timers
#SBATCH -N 1
#SBATCH -t 5

srun -N1 -n1 ./vector_addition
