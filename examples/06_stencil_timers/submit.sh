#!/bin/bash

#SBATCH -J 06_stencil_timers
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2101x
#SBATCH --reservation=cybercolombia

srun -N1 -n1 ./stencil
