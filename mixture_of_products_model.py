import haiku as hk
from jax.nn import softmax
import jax.numpy as jnp
from jax.random import categorical
import jax
import numpy as np


class Product(hk.Module):
    def __init__(self, cells, idx):
        super().__init__()
        self.cells = cells

    def __call__(self, t):
        weekly_marginal = hk.get_parameter(
            f'week_{t}',
            (self.cells[t],),
            init=hk.initializers.RandomNormal(),
            dtype='float32'
        )

        return softmax(weekly_marginal, axis=0)


class MixtureOfProductsModel(hk.Module):
    def __init__(self, cells, weeks, n, name="MixtureOfProductsModel", learn_weights=True):
        super().__init__(name=name)
        self.weeks = weeks
        self.cells = cells
        self.n = n  # number of product distributions
        self.products = []
        self.learn_weights = learn_weights

    def get_prod_k_marginal(self, k, components):
        prod_k_marginal = jnp.asarray(1)
        for idx in range(len(components)):
            prod_k_marginal = jnp.tensordot(prod_k_marginal, components[idx][k], axes=0)  # indexing with k should be ok now?
        return prod_k_marginal

    def get_marginal(self, weights, tsteps):
        components = [softmax(hk.get_parameter(f'week_{t}', (self.n, self.cells[t]), init=hk.initializers.RandomNormal(), dtype='float32')) for t in tsteps]
        vectorized_get_prod_k_marginal = hk.vmap(self.get_prod_k_marginal, split_rng=False, in_axes=(0, None))
        ks = jnp.arange(self.n)
        marginals = vectorized_get_prod_k_marginal(ks, components)
        shape = tuple([len(weights)] + [1 for d in range(len(tsteps))])
        marginals *= jnp.array([weights]).reshape(shape)
        marginals = marginals.sum(axis=0)
        return marginals

    def __call__(self):
        if self.learn_weights:
            # initialize weights
            weights = hk.get_parameter(
                'weights',
                (self.n,),
                init=hk.initializers.RandomNormal(),
                dtype='float32'
            )
        else:
            # fix all weights to be equal
            weights = jnp.zeros(self.n)
        weights = softmax(weights, axis=0)

        # idea: list of T jnp.arrays of dimension n x cells[t]
        # compute weekly / pairwise marginals from this list
        components = [softmax(
            hk.get_parameter(f'week_{t}', (self.n, self.cells[t]), init=hk.initializers.RandomNormal(),
                             dtype='float32')) for t in range(self.weeks)]

        # TODO: see if we can vmap this as well? (don't think we can)
        single_tstep_marginals = [self.get_marginal(weights, [t]) for t in range(self.weeks)]
        pairwise_marginals = [self.get_marginal(weights, [t, t + 1]) for t in range(self.weeks - 1)]
        return single_tstep_marginals, pairwise_marginals


def predict(cells, weeks, n, learn_weights=True):
    model = MixtureOfProductsModel(cells, weeks, n, learn_weights=learn_weights)
    return model()


model_forward = hk.transform(predict)

"""
Arguments
mop_params: parameters of a mixture of products model
tsteps: the weekly random variables in the desired marginal

Returns
marginal: the desired marginal of the mixture of products
"""
def compute_marginal(params, tsteps):
    weights = softmax(params['MixtureOfProductsModel']['weights'])
    marginal = 0
    n_products = params['MixtureOfProductsModel']['week_0'].shape[0]
    for k in range(n_products):
        prod_k_marginal = jnp.asarray(1)
        for tstep in tsteps:
            prod_k_marginal = jnp.tensordot(prod_k_marginal, softmax(params[f'MixtureOfProductsModel'][f'week_{tstep}'][k]), axes=0)
        marginal += weights[k] * prod_k_marginal
    return marginal

"""
Arguments
params: parameters of a mixture of products model
observations: a list of tuples (tstep, obs)

Returns
prob: the probability of the observations w.r.t to the MOP given by params
"""
def get_prob(params, observations):
    weights = softmax(params['MixtureOfProductsModel']['weights'])
    prob = 0
    n_products = params['MixtureOfProductsModel']['week_0'][0].shape[0]
    for k in range(n_products):
        prod_k_prob = 1
        for tstep, cell in observations:
            prod_k_prob *= softmax(params[f'MixtureOfProductsModel'][f'week_{tstep}'][k])[cell]
        prob += weights[k] * prod_k_prob
    return prob

"""
Arguments:
params: mixture of products parameters
tsteps: list of timesteps in the conditional distribution
conditions: list of tuples (timestep, cell) to condition on

Returns:
conditional: the conditional distribution over tsteps conditioned on observations
"""
def forecast(params, tsteps, conditions):
    n_products = len(params.keys()) - 1
    # compute weights pi for each of the corresponding conditionals of the product distributions
    weights = softmax(params['MixtureOfProductsModel']['weights'])
    pi = jnp.zeros(n_products)
    for r in range(n_products):
        likelihood_r = 1
        for (t, obs) in conditions:
            likelihood_r *= softmax(params[f'MixtureOfProductsModel/Product{r}'][f'week_{t}'])[obs]
        pi = pi.at[r].set(weights[r] * likelihood_r)
    pi /= pi.sum()  # normalize the weights

    # compute the final conditional, summing over the conditionals for each product k weighted by pi[k]
    conditional = 0
    for k in range(n_products):
        prod_k_conditional = jnp.asarray(1)
        for tstep in tsteps:
            prod_k_conditional = jnp.tensordot(prod_k_conditional, softmax(params[f'MixtureOfProductsModel/Product{k}'][f'week_{tstep}']), axes=0)
        conditional += pi[k] * prod_k_conditional

    return conditional

"""
Arguments
params: parameters of a mixture of products model
observations: a list of tuples (tstep, obs)
conditions: a list of tuples (tstep, obs) to condition on

Returns
conditional_prob: the probability of the observations given the conditions w.r.t to the MOP given by params
"""
def get_forecast_prob(params, observations, conditions):
    n_products = len(params.keys()) - 1
    # compute weights pi for each of the corresponding conditionals of the product distributions
    weights = softmax(params['MixtureOfProductsModel']['weights'])
    pi = jnp.zeros(n_products)
    for r in range(n_products):
        likelihood_r = 1
        for (t, obs) in conditions:
            likelihood_r *= softmax(params[f'MixtureOfProductsModel/Product{r}'][f'week_{t}'])[obs]
        pi = pi.at[r].set(weights[r] * likelihood_r)
    pi /= pi.sum()  # normalize the weights

    # compute the final conditional, summing over the conditionals for each product k weighted by pi[k]
    conditional_prob = 0
    for k in range(n_products):
        prod_k_conditional_prob = jnp.asarray(1)
        for tstep, cell in observations:
            prod_k_conditional_prob *= softmax(params[f'MixtureOfProductsModel/Product{k}'][f'week_{tstep}'])[cell]
        conditional_prob += pi[k] * prod_k_conditional_prob

    return conditional_prob
"""
Arguments: 
params: mixture of products parameters
Returns
route: a T-timestep route sampled from the mixture of products model
"""
def sample_route(params):
    weight_logits = jnp.asarray(params['MixtureOfProductsModel']['weights'], dtype=float)
    key = hk.PRNGSequence(np.random.randint(100))
    k = categorical(next(key), weight_logits)
    route = []
    T = len(params['MixtureOfProductsModel/Product0'].keys())
    for t in range(T):
        route.append(categorical(next(key), params[f'MixtureOfProductsModel/Product{k}'][f'week_{t}']))
    return route

"""
Arguments:
params: mixture of products parameters
timesteps_to_sample: list of timesteps to sample
observations: list of tuples (timestep, observation) to condition on

Returns:
conditional_sample: a list where element i is an observation at timesteps_to_sample[i], conditioned on observations
"""
def sample_locations_conditional(params, timesteps_to_sample, observations):
    weights = softmax(params['MixtureOfProductsModel']['weights'])
    n_products = len(params.keys()) - 1
    pi = jnp.zeros(n_products)
    for r in range(n_products):
        likelihood_r = 1
        for (t, obs) in observations:
            likelihood_r *= softmax(params[f'MixtureOfProductsModel/Product{r}'][f'week_{t}'])[obs]
        pi[r] = weights[r] * likelihood_r
    pi /= pi.sum() # normalize the weights
    key = hk.PRNGSequence(np.random.randint(100))
    k = categorical(next(key), jnp.log(pi))
    conditional_sample = []
    for t in timesteps_to_sample:
        conditional_sample.append(categorical(next(key), params[f'MixtureOfProductsModel/Product{k}'][f'week_{t}']))
    return conditional_sample

