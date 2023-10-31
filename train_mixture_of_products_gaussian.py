from functools import partial
from mixture_of_products_model_training import Datatuple, mask_input
from mixture_of_products_model_training_gaussian import loss_fn, train_model
import time
import pickle
import argparse
import os
import h5py
import numpy as np
import haiku as hk
import optax
from equinox import filter_jit
from jax.random import PRNGKey

print = partial(print, flush=True)

parser = argparse.ArgumentParser(description='Run an amewoo model')
parser.add_argument('out_dir', type=str, help='directory to save model params and losses')
parser.add_argument('params_path', type=str, help='path to scales and centers .pkl file')
parser.add_argument('--species', type=str, help='species name', default="amewoo")
parser.add_argument('--resolution', type=int, help='model resolution', default=48)
parser.add_argument('--root', type=str, help='hdf root directory', default="/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_models")
parser.add_argument('--coords_dir', type=str, help='directory containing coords lookup array', default="/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/experiments/gaussian_mop_coords_lists/")
parser.add_argument('--obs', help='Weight on the observation term of the loss', default=1.0, type=float)
parser.add_argument('--dw', help='Weight on the distance penalty in the loss', default=1e-2, type=float)
parser.add_argument('--ew', help='Weight on the joint entropy of the model', default=1e-4, type=float)
parser.add_argument('--dp', help='The exponent of the distance penalty', default=0.4, type=float)
parser.add_argument("--dont_normalize", action="store_true", help="don't normalize distance matrix")
parser.add_argument('--learning_rate', help='Learning rate for Adam optimizer', default=0.1, type=float)
parser.add_argument('--training_steps', help='The number of training iterations', default=600, type=int)
parser.add_argument('--rng_seed', help='Random number generator seed', default=17, type=int)
parser.add_argument('--n', help='Number of mixture components', default=10, type=int)

args = parser.parse_args()

print(str(args))

hdf_src = os.path.join(args.root, f'{args.species}_2021_{args.resolution}km.hdf5')
file = h5py.File(hdf_src, 'r')

true_densities = np.asarray(file['distr']).T

weeks = true_densities.shape[0]
total_cells = true_densities.shape[1]

distance_vector = np.asarray(file['distances'])**args.dp
if not args.dont_normalize:
    distance_vector *= 1 / (100**args.dp)
masks = np.asarray(file['geom']['dynamic_mask']).T.astype(bool)

dtuple = Datatuple(weeks, total_cells, distance_vector, masks)
distance_matrices, masked_densities = mask_input(true_densities, dtuple)
cells = [d.shape[0] for d in masked_densities]
#distance_matrices, masked_densities = pad_input(distance_matrices, masked_densities, cells)

# load the coords lookup list & scales, and centers arrays
with open(os.path.join(args.coords_dir, f'{args.species}_2021_{args.resolution}km.pkl'), 'rb') as f:
    coords = pickle.load(f)

# load in the coords and centers
with open(os.path.join(args.params_path), 'rb') as f:
    initial_params_dict = pickle.load(f)
    centers = initial_params_dict['centers']
    scales = initial_params_dict['scales']
    weights = initial_params_dict['weights']

# Get the random seed and optimizer
key = PRNGKey(args.rng_seed)
optimizer = optax.adam(args.learning_rate)

# Instantiate loss function
loss_fn = filter_jit(partial(loss_fn,
                      true_densities=masked_densities,
                      d_matrices=distance_matrices,
                      obs_weight=1,
                      dist_weight=1e-2,
                      ent_weight=1e-4))

t1 = time.time()
# Run Training and get params and losses
params, loss_dict = train_model(loss_fn,
                                optimizer,
                                args.training_steps,
                                args.n,
                                dtuple.weeks,
                                coords,
                                scales,
                                centers,
                                weights,
                                key)
print(f"training took {((time.time() - t1) / 60):.4f} minutes")

# save everything to a file in save_dir, save metadata
metadata_str = f'{args.species}_{args.resolution}km_obs{args.obs}_ent{args.ew}_dist{args.dw}_pow{args.dp}_n{args.n}_key{args.rng_seed}'
metadata_obj = vars(args)
with open(os.path.join(args.out_dir, f'mop_params_{metadata_str}.pkl'), 'wb') as fout:
    pickle.dump({'params': params, 'metadata': metadata_obj}, fout)
with open(os.path.join(args.out_dir, f'mop_losses_{metadata_str}.pkl'), 'wb') as fout:
    pickle.dump({'losses': loss_dict, 'metadata': metadata_obj}, fout)
