#!/bin/bash

#SBATCH -J 04_stencil
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2104x

srun -N1 -n1 ./stencil
