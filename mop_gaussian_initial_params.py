from jax.random import categorical
import jax.numpy as jnp
import pickle
import argparse
import haiku as hk
import os

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

parser = argparse.ArgumentParser(description='Run an amewoo model')
parser.add_argument('out_dir', type=str, help='directory to save model params and losses')
parser.add_argument('scale', type=float, help='scale to use for all marginals')
parser.add_argument('--resolution', type=int, help='model resolution', default=48)
parser.add_argument('--species', type=str, help='species name', default="amewoo")
parser.add_argument('--markov_dir', type=str, help='path to markov chain directory', default='/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/experiments/markov_chain_baselines')
parser.add_argument('--coords_dir', type=str, help='directory containing coords lookup array', default="/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/mixture_of_products/experiments/gaussian_mop_coords_lists/")
parser.add_argument('--obs', help='Weight on the observation term of the loss', default=1.0, type=float)
parser.add_argument('--dw', help='Weight on the distance penalty in the loss', default=1e-2, type=float)
parser.add_argument('--ew', help='Weight on the joint entropy of the model', default=1e-4, type=float)
parser.add_argument('--dp', help='The exponent of the distance penalty', default=0.4, type=float)
parser.add_argument('--rng_seed', help='Random number generator seed', default=17, type=int)
parser.add_argument('--n', help='Number of mixture components', default=1000, type=int)
parser.add_argument('--T', help='number of timesteps', default=53, type=int)

args = parser.parse_args()

# load markov chain
with open(os.path.join(args.markov_dir, f"markov_params_{args.species}_{args.resolution}_obs{args.obs}_ent{args.ew}_dist{args.dw}_pow{args.dp}.pkl"), 'rb') as f:
    markov_params = pickle.load(f)

# load the coords lookup list
with open(os.path.join(args.coords_dir, f'{args.species}_2021_{args.resolution}km.pkl'), 'rb') as f:
    coords = pickle.load(f)

# sample n routes from the markov chain, each route becomes a component
key = hk.PRNGSequence(args.rng_seed)
centers = jnp.empty((args.T, args.n, 2))
for k in range(args.n):
    centers_k = jnp.array(list(map(lambda tup: coords[tup[0]][tup[1]], enumerate(sample_trajectory(key, markov_params)))))
    centers = centers.at[:, k, :].set(centers_k)

# assemble initial params object
initial_params = {"centers": centers, "scales": args.scale*jnp.ones((args.T, args.n)), "weights": jnp.ones((args.n,))}

# save to file
fname = f"mop_gaussian_initial_params_ew{args.ew}_dw{args.dw}_dp{args.dp}_n{args.n}_scale{args.scale}.pkl"
with open(os.path.join(args.out_dir, fname), 'wb') as f:
    pickle.dump(initial_params, f)