from functools import partial
from mixture_of_products_model_training import loss_fn, train_model, Datatuple, mask_input, pad_input
import time
import pickle
import argparse
import os
import h5py
import numpy as np
from jax import jit
import haiku as hk
import optax

print = partial(print, flush=True)

parser = argparse.ArgumentParser(description='Run an amewoo model')
parser.add_argument('root', type=str, help='hdf root directory')
parser.add_argument('save_dir', type=str, help='directory to save model params and losses')
parser.add_argument('species', type=str, help='species name')
parser.add_argument('resolution', type=int, help='model resolution')
parser.add_argument('--obs_weight', help='Weight on the observation term of the loss', default=1.0, type=float)
parser.add_argument('--dist_weight', help='Weight on the distance penalty in the loss', default=1e-2, type=float)
parser.add_argument('--ent_weight', help='Weight on the joint entropy of the model', default=1e-4, type=float)
parser.add_argument('--dist_pow', help='The exponent of the distance penalty', default=0.4, type=float)
parser.add_argument("--dont_normalize", action="store_true", help="don't normalize distance matrix")
parser.add_argument('--learning_rate', help='Learning rate for Adam optimizer', default=0.1, type=float)
parser.add_argument('--training_steps', help='The number of training iterations', default=600, type=int)
parser.add_argument('--rng_seed', help='Random number generator seed', default=17, type=int)
parser.add_argument('--num_components', help='Number of mixture components', default=10, type=int)
parser.add_argument('--fix_weights', action="store_true", help="Don't learn the weights, rather, fix them to be equal")
parser.add_argument('--initialize_from_params', action="store_true", help="Initialize MoP from given parameters")
parser.add_argument('--initial_params_path', type=str, help='path to pkl with initial parameters')
args = parser.parse_args()

print(str(args))

hdf_src = os.path.join(args.root, f'{args.species}_2021_{args.resolution}km.hdf5')
file = h5py.File(hdf_src, 'r')

true_densities = np.asarray(file['distr']).T

weeks = true_densities.shape[0]
total_cells = true_densities.shape[1]

distance_vector = np.asarray(file['distances'])**args.dist_pow
if not args.dont_normalize:
    distance_vector *= 1 / (100**args.dist_pow)
masks = np.asarray(file['geom']['dynamic_mask']).T.astype(bool)

dtuple = Datatuple(weeks, total_cells, distance_vector, masks)
distance_matrices, masked_densities = mask_input(true_densities, dtuple)
cells = [d.shape[0] for d in masked_densities]
distance_matrices, masked_densities = pad_input(distance_matrices, masked_densities, cells)

# Get the random seed and optimizer
key = hk.PRNGSequence(args.rng_seed)
optimizer = optax.adam(args.learning_rate)

# Instantiate loss function
loss_fn = jit(partial(loss_fn,
                      cells=cells,
                      true_densities=masked_densities,
                      d_matrices=distance_matrices,
                      obs_weight=args.obs_weight,
                      dist_weight=args.dist_weight,
                      ent_weight=args.ent_weight,
                      num_products=args.num_components,
                      learn_weights=not args.fix_weights))

# get initial params if applicable
initial_params = None
initial_params_metadata = ""
if args.initialize_from_params:
    with open(args.initial_params_path, 'rb') as f:
        params_obj = pickle.load(f)
        initial_params = params_obj["params"]
        initial_params_metadata = f"_dim{params_obj['radius']}_scale{params_obj['scale']}"
t1 = time.time()
# Run Training and get params and losses
params, loss_dict = train_model(loss_fn,
                                optimizer,
                                args.training_steps,
                                cells,
                                dtuple.weeks,
                                key,
                                num_products=args.num_components,
                                learn_weights=not args.fix_weights,
                                initial_params=initial_params)
print(f"training took {((time.time() - t1) / 60):.4f} minutes")

# save everything to a file in save_dir
metadata = f'{args.species}_{args.resolution}km_obs{args.obs_weight}_ent{args.ent_weight}_dist{args.dist_weight}_pow{args.dist_pow}_n{args.num_components}_key{args.rng_seed}' + initial_params_metadata
if args.fix_weights:
    metadata += "_fixed_weights"
with open(os.path.join(args.save_dir, f'mop_params_{metadata}.pkl'), 'wb') as fout:
    pickle.dump(params, fout)
with open(os.path.join(args.save_dir, f'mop_losses_{metadata}.pkl'), 'wb') as fout:
    pickle.dump(loss_dict, fout)
