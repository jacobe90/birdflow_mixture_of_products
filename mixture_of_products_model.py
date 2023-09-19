import haiku as hk
from jax.nn import softmax
import jax.numpy as jnp
from jax.random import categorical
import jax
import numpy as np

# MoP with vmapped get_marginal, much slower for some reason?
class MixtureOfProductsModelDev(hk.Module):
    def __init__(self, cells, weeks, n, name="MixtureOfProductsModel", learn_weights=True):
        super().__init__(name=name)
        self.weeks = weeks
        self.cells = cells
        self.n = n  # number of product distributions
        self.learn_weights = learn_weights
        self.vectorized_get_prod_k_marginal = hk.vmap(self.get_prod_k_marginal, split_rng=False, in_axes=(0, None, 0))
        self.get_marginal_vectorized = hk.vmap(self.get_marginal, split_rng=False, in_axes=(None, 0))
        self.get_components_for_week_vectorized = hk.vmap(self.get_components_for_week, split_rng=False, in_axes=(0))
        self.components = jnp.array([jnp.pad(softmax(hk.get_parameter(f'week_{t}', (self.n, self.cells[t]), init=hk.initializers.RandomNormal(), dtype='float32')), ((0,0), (0, max(self.cells) - self.cells[t]))) for t in range(self.weeks)])

    def get_prod_k_marginal(self, k, components, weight):
        prod_k_marginal = jnp.asarray(1)
        # note idx is kind of a week
        for idx in range(len(components)): # would it help to eliminate the for loop here too?
            prod_k_marginal = jnp.tensordot(prod_k_marginal, components[idx][k], axes=0)
        return weight * prod_k_marginal
    
    def get_components_for_week(self, t):
        return self.components[t]
    
    def get_marginal(self, weights, tsteps):
        components = self.get_components_for_week_vectorized(tsteps)
        return self.vectorized_get_prod_k_marginal(np.arange(self.n), components, weights).sum(axis=0)

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
        
        # vmapped calculation of single / pairwise marginals
        single_tsteps = jnp.empty((self.weeks, 1)).at[:,0].set(jnp.arange(self.weeks)).astype('int32')
        single_tstep_marginals = self.get_marginal_vectorized(weights, single_tsteps)
        pairwise_tsteps = jnp.empty((self.weeks-1, 2)).at[:,0].set(jnp.arange(self.weeks-1)).at[:, 1].set(jnp.arange(1, self.weeks)).astype('int32')
        pairwise_marginals = self.get_marginal_vectorized(weights, pairwise_tsteps)
        
        # this was necessary if we have zero probability cells (which messes up the entropy calculation later, which involves taking a log)
        single_tstep_marginals += 1e-20 * jnp.ones_like(single_tstep_marginals)
        pairwise_marginals += 1e-20 * jnp.ones_like(pairwise_marginals)
        
        return single_tstep_marginals, pairwise_marginals

# currently fastest MoP training
# 250 components ~ 6 minutes
# 1000 components ~ 20 minutes
class MixtureOfProductsModelFast(hk.Module):
    def __init__(self, cells, weeks, n, name="MixtureOfProductsModel", learn_weights=True):
        super().__init__(name=name)
        self.weeks = weeks
        self.cells = cells
        self.n = n  # number of product distributions
        self.learn_weights = learn_weights
        self.vectorized_get_prod_k_marginal = hk.vmap(self.get_prod_k_marginal, split_rng=False, in_axes=(0, None, 0))
        self.batch_size = 2
        self.batched_get_marginals = hk.vmap(self.marginal_batch, split_rng=False, in_axes=(0, None, 0))
        #self.get_marginal_vectorized = hk.vmap(self.get_marginal, split_rng=False, in_axes=(None, 0))
        self.get_components_for_week_vectorized = hk.vmap(self.get_components_for_week, split_rng=False, in_axes=(0))
    def get_prod_k_marginal(self, k, components, weight):
        prod_k_marginal = jnp.asarray(1)
        # note idx is kind of a week
        for idx in range(len(components)): # would it help to eliminate the for loop here too?
            prod_k_marginal = jnp.tensordot(prod_k_marginal, components[idx][k], axes=0)
        return weight * prod_k_marginal
    
    def marginal_batch(self, weights, components, ks):
        marginal = self.vectorized_get_prod_k_marginal(ks, components, weights).sum(axis=0)
        return marginal
    
    def get_components_for_week(self, t):
        return softmax(hk.get_parameter(f'week_{t}', (self.n, self.cells[t]), init=hk.initializers.RandomNormal(), dtype='float32'))
    
    def get_marginal(self, weights, tsteps):
        #components = self.get_components_for_week_vectorized(tsteps)
        components = [softmax(hk.get_parameter(f'week_{t}', (self.n, self.cells[t]), init=hk.initializers.RandomNormal(), dtype='float32')) for t in tsteps]

        # unbatched marginal computation
        return self.vectorized_get_prod_k_marginal(np.arange(self.n), components, weights).sum(axis=0) + 1e-20

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
        
        single_tstep_marginals = [self.get_marginal(weights, [t]) for t in range(self.weeks)]
        pairwise_marginals = [self.get_marginal(weights, [t, t + 1]) for t in range(self.weeks - 1)]
        return single_tstep_marginals, pairwise_marginals


class MixtureOfProductsModelTakeThree(hk.Module):
    def __init__(self, cells, weeks, n, name="MixtureOfProductsModel", learn_weights=True):
        super().__init__(name=name)
        self.weeks = weeks
        self.cells = cells
        self.n = n  # number of product distributions
        self.learn_weights = learn_weights

        # vmapped functions
        self.get_weighted_pairwise_marginals_of_components = hk.vmap(self.get_weighted_pairwise_marginal_of_component_k, split_rng=False, in_axes=(None, 0))
        self.get_single_timestep_marginals = hk.vmap(self.get_single_timestep_marginal, split_rng=False, in_axes=(0))
        self.get_pairwise_marginals = hk.vmap(self.get_pairwise_marginal, split_rng=False, in_axes=(0))

        # initialize parameters
        self.components = jnp.array([jnp.pad(softmax(
            hk.get_parameter(f'week_{t}', (self.n, self.cells[t]), init=hk.initializers.RandomNormal(),
                             dtype='float32')), ((0, 0), (0, max(self.cells) - self.cells[t]))) for t in
                                     range(self.weeks)])
        if self.learn_weights:
            # initialize weights
            self.weights = hk.get_parameter(
                'weights',
                (self.n,),
                init=hk.initializers.RandomNormal(),
                dtype='float32'
            )
        else:
            # fix all weights to be equal
            self.weights = jnp.zeros(self.n)
        self.weights = softmax(self.weights, axis=0)

    def get_single_timestep_marginal(self, t):
        return (self.components[t] * jnp.array([self.weights]).T).sum(axis=0)

    def get_weighted_pairwise_marginal_of_component_k(self, t, k):
        return self.weights[k] * jnp.tensordot(self.components[t][k], self.components[t+1][k], axes=0)

    def get_pairwise_marginal(self, t):
        return self.get_weighted_pairwise_marginals_of_components(t, jnp.arange(self.n)).sum(axis=0)

    def __call__(self):
        # vmapped calculation of single / pairwise marginals
        print(f"first single-tstep marginal is: {self.get_single_timestep_marginal(0)}")
        single_tsteps = jnp.arange(self.weeks, dtype='int32')
        single_tstep_marginals = self.get_single_timestep_marginals(single_tsteps)
        pairwise_tsteps = jnp.arange(self.weeks-1, dtype='int32')
        pairwise_marginals = self.get_pairwise_marginals(pairwise_tsteps)

        # this was necessary if we have zero probability cells (which messes up the entropy calculation later, which involves taking a log)
        single_tstep_marginals += 1e-20 * jnp.ones_like(single_tstep_marginals)
        pairwise_marginals += 1e-20 * jnp.ones_like(pairwise_marginals)

        return single_tstep_marginals, pairwise_marginals


def predict(cells, weeks, n, learn_weights=True):
    model = MixtureOfProductsModelTakeThree(cells, weeks, n, learn_weights=learn_weights)
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
    weights = softmax(params['MixtureOfProductsModel']['weights'])
    key = hk.PRNGSequence(np.random.randint(100))
    k = categorical(next(key), weights)
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

