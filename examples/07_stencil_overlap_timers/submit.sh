#!/bin/bash

#SBATCH -J 07_stencil_overlap_timers
#SBATCH -N 1
#SBATCH -t 5
#SBATCH --reservation=sc23

srun -N1 -n1 ./stencil
