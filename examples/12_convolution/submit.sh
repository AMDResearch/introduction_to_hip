#!/bin/bash

#SBATCH -J 09_convolution
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2104x

srun -N1 -n1 ./convolution
