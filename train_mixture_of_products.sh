#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=12GB
#SBATCH -p gypsum-1080ti
#SBATCH -G 1
#SBATCH -t 1:00:00
#SBATCH -o slurm-%x.%j.out

FIX='false'
while getopts :e:d:p:n:s:r:o:i:k:f flag
do
    case "${flag}" in
        e) ENTWEIGHT=${OPTARG};;
        d) DISTWEIGHT=${OPTARG};;
        p) DISTPOW=${OPTARG};;
        n) NCOMPONENTS=${OPTARG};;
        s) SPECIES=${OPTARG};;
        r) RES=${OPTARG};;
        o) ROOT=${OPTARG};;
        i) SAVEDIR=${OPTARG};;
        k) KEYSEED=${OPTARG};;
        f) FIX='true';;
        *) echo "invalid command: no parameter included with argument $OPTARG";;
    esac
done

if ${FIX}; then
    /home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python train_mixture_of_products.py $ROOT $SAVEDIR $SPECIES $RES --ent_weight $ENTWEIGHT --dist_weight $DISTWEIGHT --dist_pow $DISTPOW --num_components $NCOMPONENTS --rng_seed $KEYSEED --fix_weights
else
    /home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python train_mixture_of_products.py $ROOT $SAVEDIR $SPECIES $RES --ent_weight $ENTWEIGHT --dist_weight $DISTWEIGHT --dist_pow $DISTPOW --num_components $NCOMPONENTS --rng_seed $KEYSEED
fi