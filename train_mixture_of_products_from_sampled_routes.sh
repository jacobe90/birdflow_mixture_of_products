#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=15GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1:00:00
#SBATCH -o slurm-%x.%j.out

UNBOX='false'
while getopts :r:d:s:e:w:p:i:u flag
do
    case "${flag}" in
        r) ROUTES=${OPTARG};;
        d) DIM=${OPTARG};;
        s) SCALE=${OPTARG};;
        e) ENT_W=${OPTARG};;
        w) DIST_W=${OPTARG};;
        p) DIST_P=${OPTARG};;
        i) SAVE_DIR=${OPTARG};;
        u) UNBOX='true';;
        *) echo "invalid command: no parameter included with argument $OPTARG";;
    esac
done

if ${UNBOX}; then
    /home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python mixture_of_products_from_sampled_routes.py /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_models /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/experiments/markov_chain_baselines $SAVE_DIR $ROUTES $DIM $SCALE --ent_weight $ENT_W --dist_weight $DIST_W --dist_pow $DIST_P --unbox
else
    /home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python mixture_of_products_from_sampled_routes.py /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_models /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/experiments/markov_chain_baselines $SAVE_DIR $ROUTES $DIM $SCALE --ent_weight $ENT_W --dist_weight $DIST_W --dist_pow $DIST_P
fi