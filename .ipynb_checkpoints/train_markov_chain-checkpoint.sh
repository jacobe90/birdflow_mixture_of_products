#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=10GB
#SBATCH -p gypsum-1080ti
#SBATCH -G 1
#SBATCH -t 0:30:00
#SBATCH -o slurm-%x.%j.out

while getopts :e:d:p:s:r:o:i: flag
do
    case "${flag}" in
        e) ENTWEIGHT=${OPTARG};;
        d) DISTWEIGHT=${OPTARG};;
        p) DISTPOW=${OPTARG};;
        s) SPECIES=${OPTARG};;
        r) RES=${OPTARG};;
        o) ROOT=${OPTARG};;
        i) SAVEDIR=${OPTARG};;
        *) echo "invalid command: no parameter included with argument $OPTARG";;
    esac
done
/home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/BirdFlowPy/update_hdf.py $ROOT $SAVEDIR $SPECIES $RES --dist_weight $DISTWEIGHT --ent_weight $ENTWEIGHT --dist_pow $DISTPOW --dont_save_hdf 
