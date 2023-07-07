#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=12GB
#SBATCH -p gypsum-titanx
#SBATCH -G 1
#SBATCH -t 1:00:00
#SBATCH -o slurm-%x.%j.out

/home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python tests.py TestValidation.test_track_log_likelihood_works