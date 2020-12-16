#!/bin/bash
#SBATCH --job-name=sirs_simul
#SBATCH --output=sirs_simulation.out
#SBATCH --error=sirs_simulation.err
#SBATCH --time=24:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000

module load python
python sirs_simulation.py