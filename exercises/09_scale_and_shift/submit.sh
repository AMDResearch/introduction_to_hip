#!/bin/bash

#SBATCH -J 09_scale_and_shift
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2101x
#SBATCH --reservation=cybercolombia

srun -N1 -n1 ./scale_and_shift
