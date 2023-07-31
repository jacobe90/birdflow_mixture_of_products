#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=20GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 24:00:00
#SBATCH -o slurm-%x.%j.out

/home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python experiments.py