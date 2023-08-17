#TODO: change from box_radius to the more intuitive box_dim

from functools import partial
from mixture_of_products_model_training import loss_fn, train_model, Datatuple, mask_input, pad_input
import pickle
import argparse
import os
import h5py
import numpy as np
import jax.numpy as jnp
from jax.random import categorical
from scipy.stats import multivariate_normal
from jax import jit
import haiku as hk
import math
import time


def sample_trajectory(rng_seq, flow_params, ipos=None, start=1, end=None):
    if end:
        end = end
    else:
        end = len(flow_params)

    if ipos:
        pos = ipos
    else:
        init_p = flow_params['Flow_Model/Initial_Params']['z0']
        pos = categorical(next(rng_seq), init_p)

    trajectory = [int(pos)]

    for week in range(start, end):
        trans_p = flow_params[f'Flow_Model/Week_{week}']['z'][pos, :]
        pos = categorical(next(rng_seq), trans_p)
        trajectory.append(int(pos))
    return trajectory


def get_index_in_bigger_grid(cell, mask):
    true_count = -1
    new_cell = None
    for i, b in enumerate(mask):
        if b:
            true_count += 1
        if true_count == cell:
            new_cell = i
            break
    return new_cell


"""
cell: cell in grid of size len(mask)
mask: boolean array where true cells correspond to cells of a smaller grid
Returns: None if mask[cell] == False, index of cell in smaller grid (equals the number of True values in the mask before index cell)
"""


def get_index_in_smaller_grid(cell, mask):
    true_idx = -1
    if cell == None:
        return None
    if not mask[cell]:
        return None
    new_cell = None
    for i, b in enumerate(mask):
        if b:
            true_idx += 1
        if i == cell:
            new_cell = true_idx
            break
    return new_cell

"""
Arguments:
nan_mask, weekly_mask
Returns:
conversion_dict that satisfies conversion_dict[cell_in_overall_grid] = cell_in_weekly_grid or None
"""
def get_overall_to_weekly_mask_conversion_dict(nan_mask, weekly_mask):
    overall_dict = {}
    idx = 0
    for i, b in enumerate(nan_mask):
        if b:
            overall_dict[i] = idx
            idx += 1
        if not b:
            overall_dict[i] = None
    weekly_dict = {None: None}
    idx = 0
    for i, b in enumerate(weekly_mask):
        if b:
            weekly_dict[i] = idx
            idx += 1
        if not b:
            weekly_dict[i] = None
    conversion_dict = {}
    for i in range(len(nan_mask)):
        conversion_dict[i] = weekly_dict[overall_dict[i]]
    return conversion_dict


def cell_to_xy(cell, x_dim, y_dim):
    x = cell % x_dim
    y = int(cell / x_dim)
    return x, y


def xy_to_cell(x, y, x_dim, y_dim):
    cell = y * x_dim + x
    return cell


"""
cell: grid cell index (center of the box)
week: week of the cell
masks: list of dynamic masks (we care about masks[week])
nan_mask: big mask used to eliminate oceans
x_dim, y_dim: dimensions of the overall grid
box_dim: number of cells from box center to edge (not counting the center)

Returns: box, a dictionary of idx, coords pairs, where idx is a cell index for given week, coords is a tuple of the coordinates for that cell 
"""


def get_box(cell, week, masks, nan_mask, x_dim, y_dim, box_dim, conversion_dict):
    if box_dim % 2 == 0 and box_dim > 2:
        raise Exception("even box dimension > 2 unsupported")

    # if we are in the 1x1 case, just return the original cell as the center of a box
    if box_dim == 1:
        return {cell: (0, 0)}

    # grid cell index in weekly grid -> grid cell index in medium grid -> grid cell index in big grid
    medium_cell = get_index_in_bigger_grid(cell, masks[week])
    big_cell = get_index_in_bigger_grid(medium_cell, nan_mask)
    x_c, y_c = cell_to_xy(big_cell, x_dim, y_dim)
    box_r = int(box_dim / 2)
    box = {}

    if box_dim == 2:
        for y in range(y_c, min(y_dim, y_c + box_dim)):
            for x in range(x_c, min(x_dim, x_c + box_dim)):
                big_xy_cell = xy_to_cell(x, y, x_dim, y_dim)
                small_xy_cell = conversion_dict[big_xy_cell]
                if small_xy_cell is not None:
                    box[small_xy_cell] = (x - x_c, y - y_c)
        return box

    for y in range(max(0, y_c - box_r), min(y_dim, y_c + box_r + 1)):
        for x in range(max(0, x_c - box_r), min(x_dim, x_c + box_r + 1)):
            # convert (x,y) in big grid -> cell in the big grid
            big_xy_cell = xy_to_cell(x, y, x_dim, y_dim)

            # convert big_xy_cell back to a cell in the small grid for week
            # store coordinates in box
            small_xy_cell = conversion_dict[big_xy_cell]
            if small_xy_cell is not None:
                box[small_xy_cell] = (x - x_c, y - y_c)
    return box


"""
Arguments:
box: the box to apply the mvn to
week: the week of the marginal

Returns: a vector such that applying softmax yields marginal with zeroes everywhere outside of the indices given by box (whose indices have probability values given by an MVN)
"""


def get_weekly_marginal(box_center, week, cells, masks, nan_mask, x_dim, y_dim, box_dim, conversion_dict):
    box = get_box(box_center, week, masks, nan_mask, x_dim, y_dim, box_dim, conversion_dict)
    marginal = np.empty(cells[week])
    marginal.fill(-jnp.inf)
    for idx, coords in box.items():
        marginal[idx] = math.log(multivariate_normal.pdf(coords, mean=[0, 0], cov=[[2, 0], [0, 2]]))
    return jnp.array(marginal)


"""
Arguments:
routes: An n x T array of sampled routes (n sampled routes of T timesteps)
Returns:
mixture of products parameters, each components corresponds to one of the routes
"""


def mop_from_routes(routes, cells, masks, nan_mask, x_dim, y_dim, box_dim):
    n = routes.shape[0]
    mop_params = {'MixtureOfProductsModel': {'weights': jnp.ones(n)}}
    T = routes.shape[1]
    for t in range(T):
        week_t_components = jnp.empty((n, cells[t]))
        conversion_dict = get_overall_to_weekly_mask_conversion_dict(nan_mask, masks[t])
        for k in range(n):
            week_t_components = week_t_components.at[k, :].set(
                get_weekly_marginal(routes[k][t], t, cells, masks, nan_mask, x_dim, y_dim, box_dim, conversion_dict))
        mop_params['MixtureOfProductsModel'][f'week_{t}'] = week_t_components
    return mop_params


# print = partial(print, flush=True)
#
# parser = argparse.ArgumentParser(description='Run an amewoo model')
# parser.add_argument('root', type=str, help='directory containing hdf5 info')
# parser.add_argument('markov_params_dir', type=str, help='directory containing markov chain parameters')
# parser.add_argument('save_dir', type=str, help='directory to save model params and losses')
# parser.add_argument('num_routes', type=int,
#                     help='number of routes to sample from markov chain / number of components in the final mixture-of-products model')
# parser.add_argument('box_dim', type=int, help='length of each marginal box edge')
# parser.add_argument('scale', type=float, help='scale of gaussian used to generate proabilities')
# parser.add_argument('--species', type=str, help='species name', default='amewoo')
# parser.add_argument('--resolution', type=int, help='model resolution', default=48)
# parser.add_argument('--obs_weight', help='Weight on the observation term of the loss', default=1.0, type=float)
# parser.add_argument('--dist_weight', help='Weight on the distance penalty in the loss', default=0.01, type=float)
# parser.add_argument('--ent_weight', help='Weight on the joint entropy of the model', default=0.0001, type=float)
# parser.add_argument('--dist_pow', help='The exponent of the distance penalty', default=0.4, type=float)
# parser.add_argument("--dont_normalize", action="store_true", help="don't normalize distance matrix")
# parser.add_argument('--rng_seed', help='Random number generator seed', default=17, type=int)
# args = parser.parse_args()
# print(args)
# t1 = time.time()
# # load all hdf5 info
# hdf_src = os.path.join(args.root, f'{args.species}_2021_{args.resolution}km.hdf5')
# file = h5py.File(hdf_src, 'r')
# true_densities = np.asarray(file['distr']).T
# weeks = true_densities.shape[0]
# total_cells = true_densities.shape[1]
#
# # create distance vector
# distance_vector = np.asarray(file['distances']) ** args.dist_pow
# if not args.dont_normalize:
#     distance_vector *= 1 / (100 ** args.dist_pow)
#
# # create and pad distance matrices and masked densities, load nan_mask
# masks = np.asarray(file['geom']['dynamic_mask']).T.astype(bool)
# dtuple = Datatuple(weeks, total_cells, distance_vector, masks)
# distance_matrices, masked_densities = mask_input(true_densities, dtuple)
# cells = [d.shape[0] for d in masked_densities]
# distance_matrices, masked_densities = pad_input(distance_matrices, masked_densities, cells)
# nan_mask = np.asarray(file['geom']['mask']).flatten().astype(bool)
#
# # get x / y dimensions of the grid
# x_dim = int(np.asarray(file['geom']['ncol']))
# y_dim = int(np.asarray(file['geom']['nrow']))
#
# # Get the random seed
# key = hk.PRNGSequence(args.rng_seed)
#
# # load markov chain and sample routes
# with open(os.path.join(args.markov_params_dir,   f'markov_params_{args.species}_{args.resolution}_obs{args.obs_weight}_ent{args.ent_weight}_dist{args.dist_weight}_pow{args.dist_pow}.pkl'),'rb') as f:
#     markov_params = pickle.load(f)
# routes = np.asarray([sample_trajectory(key, markov_params) for k in range(args.num_routes)])
# print(f"preprocessing: {(time.time()-t1)/60:.4f} min")
#
# t2 = time.time()
# # generate mixture of products parameters from sampled routes
# mop_params = mop_from_routes(routes, cells, masks, nan_mask, x_dim, y_dim, args.box_radius)
# print(f"generating parameters: {(time.time()-t2)/60:.4f} min")
#
# t3 = time.time()
# # evaluate loss function
# loss = jit(partial(loss_fn, cells=cells,
#                    true_densities=masked_densities,
#                    d_matrices=distance_matrices,
#                    obs_weight=args.obs_weight,
#                    dist_weight=args.dist_weight,
#                    ent_weight=args.ent_weight,
#                    num_products=args.num_routes))(mop_params)
# print(f"evaluating loss function: {(time.time()-t3)/60:.4f} min")
#
# # save parameters and loss
# with open(os.path.join(args.save_dir,         f'{args.species}_mop_from_routes_params_and_losses_{args.resolution}_obs{args.obs_weight}_ent{args.ent_weight}_dist{args.dist_weight}_pow{args.dist_pow}_radius{args.box_radius}_n{args.num_routes}.pkl'),
#           'wb') as f:
#     pickle.dump({'n': args.num_routes, 'radius': args.box_radius, 'params': mop_params, 'losses': loss}, f)
