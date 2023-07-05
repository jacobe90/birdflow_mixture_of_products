#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=12GB
#SBATCH -p gypsum-titanx
#SBATCH -G 1
#SBATCH -t 1:00:00
#SBATCH -o slurm-%x.%j.out

while getopts e:d:p:s:r:o: flag
do
    case "${flag}" in
        e) ENTWEIGHT=${OPTARG};;
        d) DISTWEIGHT=${OPTARG};;
        p) DISTPOW=${OPTARG};;
        s) SPECIES=${OPTARG};;
        r) RES=${OPTARG};;
        o) ROOT=${OPTARG};;
    esac
done

/home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python train_mixture_of_products.py $ROOT $SPECIES $RES --ent_weight ENTWEIGHT --dist_weight DISTWEIGHT --dist_pow DISTPOW