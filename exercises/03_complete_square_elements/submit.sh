#!/bin/bash

#SBATCH -J 03_complete_square_elements
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -p mi2101x
#SBATCH --reservation=cybercolombia

srun -N1 -n1 ./square_elements
