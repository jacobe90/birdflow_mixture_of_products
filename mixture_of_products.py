import numpy as np

"""
Arguments:
n: number of product distributions in the mixture
T: total number of timesteps
locations: array of size T. locations[t] := number of locations a bird can be at timestep t

Returns:
mixture of products parameters
{weights: [array of size n], products: [array of size n, products[k]:= [array of size T, products[k][t]:= 
array of size locations[t] that parametrizes the weekly marginal of product distribution k for week t] ]}
"""
def mixture_of_products_params(n, T, locations):
    params = {'n': n, 'T': T, 'locations': locations, 'weights': np.random.rand(n), 'products': []}
    for k in range(n):
        params['products'].append([np.random.rand(locations[t]) for t in range(T)])
    return params

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sample_categorical(weights):
    return np.where(np.random.multinomial(n=1, pvals=weights) == 1)[0][0]

"""
Compute an arbitrary marginal distribution of the mixture of products
Arguments:
params: mixture of products parameters
tsteps: vector of timesteps [t_1, ..., t_j] we wish to compute a marginal for. timesteps cannot be repeated, and cannot exceed params['n']

Returns:
marginal:= a tensor of dimensions params[locations[t_1]...locations[t_j]] that encodes a marginal.
marginal[l_1, ..., l_j] := p(x_(t_1) = l_1, ..., x_(t_j)=l_j)
"""
def compute_marginal(params, tsteps):
    weights = softmax(params['weights'])
    dims = tuple(map(lambda tup: tup[1], filter(lambda tup: tup[0] in tsteps, list(enumerate(params['locations'])))))
    marginal = np.zeros(dims)
    for k in range(params['n']):
        prod_k_marginal = np.asarray(1)
        for tstep in tsteps:
            prod_k_marginal = np.tensordot(prod_k_marginal, softmax(params['products'][k][tstep]), axes=0)
        marginal += weights[k] * prod_k_marginal
    return marginal


def forecast(params, tsteps, observations):
    pass


def sample_route(params):
    weights = softmax(params['weights'])
    k = sample_categorical(weights)
    route = []
    for t in range(params['T']):
        route.append(sample_categorical(softmax(params['products'][k][t])))
    return route

def sample_locations_conditional(params, locations_to_sample, observations):
    pass