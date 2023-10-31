#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=15GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1:00:00
#SBATCH -o slurm-%x.%j.out

while getopts :e:d:p:n:s:o: flag
do
    case "${flag}" in
        s) SCALE=${OPTARG};;
        o) OUTDIR=${OPTARG};;
        e) ENTWEIGHT=${OPTARG};;
        d) DISTWEIGHT=${OPTARG};;
        p) DISTPOW=${OPTARG};;
        n) NCOMPONENTS=${OPTARG};;
        *) echo "invalid command: no parameter included with argument $OPTARG";;
    esac
done

/home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python mop_gaussian_initial_params.py $OUTDIR $SCALE --dw $DISTWEIGHT --ew $ENTWEIGHT --dp $DISTPOW --n $NCOMPONENTS