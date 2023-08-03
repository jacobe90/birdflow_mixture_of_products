import haiku as hk
from jax.nn import softmax
import jax.numpy as jnp
from jax.random import categorical
import jax
import numpy as np

# changes so far:
# - vmap computation of marginals for each component (vector of components) -> vector of marginals
# - jit of the init function (eliminated all memory overflow issues I was having)
# in progress:
# - vmap computation of all pairwise and multi-timestep marginals (vector of vectors of timesteps) -> vector of marginals
# issue: can't request the parameters for the cells within a vmapped function, and can't easily precompute everything
#        and then look everything up because we can't store the cells in a jax array due to varying cells per week
# fix: pregenerate all the cell probabilities, store in a bigger homogenous JAX array with zeroes for padding

class MixtureOfProductsModel(hk.Module):
    def __init__(self, cells, weeks, n, name="MixtureOfProductsModel", learn_weights=True):
        super().__init__(name=name)
        self.weeks = weeks
        self.cells = cells
        self.n = n  # number of product distributions
        self.products = []
        self.learn_weights = learn_weights
        self.vectorized_get_prod_k_marginal = hk.vmap(self.get_prod_k_marginal, split_rng=False, in_axes=(0, None, 0))
        self.batch_size = 2
        # self.batched_get_marginals = hk.vmap(self.marginal_batch, split_rng=False, in_axes=(0, None, 0))
        self.get_marginal_vectorized = hk.vmap(self.get_marginal, split_rng=False, in_axes=(None, 0))
        self.get_components_for_week_vectorized = hk.vmap(self.get_components_for_week, split_rng=False, in_axes=(0))
        self.components = jnp.array([jnp.pad(softmax(hk.get_parameter(f'week_{t}', (self.n, self.cells[t]), init=hk.initializers.RandomNormal(), dtype='float32')), ((0,0), (0, max(self.cells) - self.cells[t]))) for t in range(self.weeks)])

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
        return self.components[t]
        #return softmax(hk.get_parameter(f'week_{t}', (self.n, self.cells[t]), init=hk.initializers.RandomNormal(), dtype='float32'))
    
    def get_marginal(self, weights, tsteps):
        components = self.get_components_for_week_vectorized(tsteps)
        #components = [softmax(hk.get_parameter(f'week_{t}', (self.n, self.cells[t]), init=hk.initializers.RandomNormal(), dtype='float32')) for t in tsteps]

        # unbatched marginal computation
        return self.vectorized_get_prod_k_marginal(np.arange(self.n), components, weights).sum(axis=0)
        
        # compute marginal (batched)
        # ks = jnp.arange(self.n)
        # batched_ks = ks[:self.batch_size*int(self.n/self.batch_size)].reshape(int(self.n/self.batch_size), self.batch_size)
        # batched_weights = weights[:self.batch_size*int(self.n/self.batch_size)].reshape(int(self.n/self.batch_size), self.batch_size)
        
        # compute weighted marginals of components in batches of 150
        #marginal = self.batched_get_marginals(batched_weights, components, batched_ks).sum(axis=0)
        
        # compute marginals of any remaining components
        # if self.n % self.batch_size != 0:
        #     marginal += self.vectorized_get_prod_k_marginal(ks[-1 * (self.n%self.batch_size):], components, weights[-1 * (self.n%self.batch_size):]).sum(axis=0)
        
        #return marginal

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

        # TODO: see if we can vmap this as well? (idea: vmap over the tsteps component of get_marginal, do two calls to this vmapped function to get all the single tstep and pairwise marginals)
        single_tsteps = jnp.empty((self.weeks, 1)).at[:,0].set(jnp.arange(self.weeks)).astype('int32')
        single_tstep_marginals = self.get_marginal_vectorized(weights, single_tsteps)
        pairwise_tsteps = jnp.empty((self.weeks-1, 2)).at[:,0].set(jnp.arange(self.weeks-1)).at[:, 1].set(jnp.arange(1, self.weeks)).astype('int32')
        pairwise_marginals = self.get_marginal_vectorized(weights, pairwise_tsteps)
        #print(self.get_components_for_week_vectorized(jnp.array([0, 1])))
        #print(self.get_marginal_vectorized(weights, jnp.array([[0, 1]])))
        # single_tstep_marginals = [self.get_marginal_vectorized(weights, jnp.array([[t]])) for t in range(self.weeks)]
        # pairwise_marginals = [self.get_marginal_vectorized(weights, jnp.array([[t, t + 1]])) for t in range(self.weeks - 1)]
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

