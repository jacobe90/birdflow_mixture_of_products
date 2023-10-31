#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=15GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1:00:00
#SBATCH -o slurm-%x.%j.out

while getopts :e:d:p:n:o:i: flag
do
    case "${flag}" in
        e) ENTWEIGHT=${OPTARG};;
        d) DISTWEIGHT=${OPTARG};;
        p) DISTPOW=${OPTARG};;
        n) NCOMPONENTS=${OPTARG};;
        o) OUTDIR=${OPTARG};;
        i) INITIAL_PARAMS_PATH=${OPTARG};;
        *) echo "invalid command: no parameter included with argument $OPTARG";;
    esac
done

/home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python train_mixture_of_products_gaussian.py $OUTDIR $INITIAL_PARAMS_PATH --ew $ENTWEIGHT --dw $DISTWEIGHT --dp $DISTPOW --n $NCOMPONENTS