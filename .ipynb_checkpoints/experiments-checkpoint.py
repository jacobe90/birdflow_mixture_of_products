import os
import numpy as np
import time
"""
Arguments:
- ent_weights: array of entropy weights to search over
- dist_weights: array of distance weights to search over
- dist_pows: array of distance powers to search over
- n_components: array of number of components to try

Runs a grid search over these values of ent_weight, dist_weight, dist_pow, and n_components
"""
def mixture_of_products_grid_search(ent_weights, dist_weights, dist_pows, n_components, save_dir=None, group_size=8):
    hypers = [(ew, dw, dp, n) for ew in ent_weights for dw in dist_weights for dp in dist_pows for n in n_components]
    # evaluate hyperparameters in groups of group_size
    job_file = "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/birdflow_mixture_of_products/train_mixture_of_products.sh"
    root_dir = "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_models"
    n_groups = int(len(hypers) / group_size)
    for i in range(n_groups):
        print(f"evaluating hyperparameters for group {i+1} of {n_groups}")
        hyper_group = hypers[group_size*i:group_size*(i+1)]
        for ew, dw, dp, n in hyper_group:
            print(f"evaluating hyperparameters ew={ew}, dw={dw}, dp={dp}, n={n}")
            os.system(f"sbatch {job_file} -e {ew} -d {dw} -p {dp} -n {n} -s amewoo -r 48 -o {root_dir} -i {save_dir} -k 42")
        time.sleep(60 * 15)
    # evaluate any remaining hyperparameters
    hyper_group = hypers[-1 * (len(hypers) % group_size):]
    print("evaluating remainder...")
    for ew, dw, dp, n in hyper_group:
        print(f"evaluating hyperparameters ew={ew}, dw={dw}, dp={dp}, n={n}")
        os.system(f"sbatch {job_file} -e {ew} -d {dw} -p {dp} -n {n} -s amewoo -r 48 -o {root_dir} -i {save_dir} -k 42")
    print("finished grid search!")


"""
Trains a mixture of products with given hyperparameter settings n_batch times (using a different initialization seed
each time), saves the output in a directory.

Goal is to look at the variability during training - do we always find the same optima?
"""
def training_variability(ent_weight, dist_weight, dist_pow, n_components, n_batch, save_dir=None, group_size=8):
    job_file = "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/birdflow_mixture_of_products/train_mixture_of_products.sh"
    root_dir = "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_models"
    print(f"training {n_batch} models with hyperparameter settings ew={ent_weight}, dw={dist_weight}, dp={dist_pow}, n={n_components}")
    for i in range(int(n_batch / group_size)):
        for j in range(group_size):
            rng_key_seed = np.random.randint(1000000)
            os.system(f"sbatch {job_file} -e {ent_weight} -d {dist_weight} -p {dist_pow} -n {n_components} -s amewoo -r 48 -o {root_dir} -i {save_dir} -k {rng_key_seed}")
        time.sleep(60 * 15)


"""
Train a mixture of products with given hyperparameter settings, fixing the weights of the components to be equal
"""
def train_with_equally_weighted_components(ent_weight, dist_weight, dist_pow, n_components, save_dir=None):
    job_file = "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/birdflow_mixture_of_products/train_mixture_of_products.sh"
    root_dir = "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_models"
    os.system(f"sbatch {job_file} -e {ent_weight} -d {dist_weight} -p {dist_pow} -n {n_components} -s amewoo -r 48 -o {root_dir} -i {save_dir} -k 42 -f")
    pass

"""
Train a markov chain with given hyperparameter settings to be used as a baseline for mixture of products training experiments
"""
def markov_chain_baseline(ent_weight, dist_weight, dist_pow, save_dir=None):
    job_file = "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/birdflow_mixture_of_products/train_markov_chain.sh"
    root_dir = "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_models"
    os.system(f"sbatch {job_file} -e {ent_weight} -d {dist_weight} -p {dist_pow} -s amewoo -r 48 -o {root_dir} -i {save_dir}")

if __name__=="__main__":
    #mixture_of_products_grid_search([1e-4], [1e-2], [0.4], [6, 8, 10, 12, 14, 16, 18, 20], save_dir="/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/experiments/number_of_components_1d_grid_search")
    #markov_chain_baseline(1e-4, 1e-2, 0.4, save_dir="/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/experiments/markov_chain_baselines")
    #train_with_equally_weighted_components(1e-4, 1e-2, 0.4, 10, save_dir="/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/experiments/fix_weights_of_components")
    training_variability(1e-4, 1e-2, 0.4, 10, 40, save_dir="/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/experiments/training_variability_10_components", group_size=8)