#!/bin/bash
#SBATCH --job-name=pde_spatial_simul
#SBATCH --output=pde_simulation.out
#SBATCH --error=pde_simulation.err
#SBATCH --time=00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000

module load python
python sir_simulation.py