#!/bin/bash

#SBATCH -J 08_matrix_addition
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2104x
#SBATCH --reservation=carla24

srun -N1 -n1 ./matrix_addition
