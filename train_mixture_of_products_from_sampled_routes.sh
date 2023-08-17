#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=15GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1:00:00
#SBATCH -o slurm-%x.%j.out

while getopts :r:d: flag
do
    case "${flag}" in
        r) ROUTES=${OPTARG};;
        d) DIM=${OPTARG};;
        *) echo "invalid command: no parameter included with argument $OPTARG";;
    esac
done

/home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python mixture_of_products_from_sampled_routes.py /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_models /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/experiments/markov_chain_baselines /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/experiments/mixture_of_products_from_sampled_routes/ $ROUTES $DIM 1