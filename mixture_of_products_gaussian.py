import equinox as eqx
import jax.numpy as jnp
from jax.nn import softmax
from jax import vmap
import jax
import math


def get_marginals_of_components_for_week(center, scale, coords):
    # compute densities from the MVN pdf
    d = jnp.linalg.norm(coords - center, axis=1) ** 2
    probs = (1 / (2 * math.pi * scale)) * jnp.exp(-(0.5 / scale) * d)

    # set densities that are below threshold to the threshold value
    probs = jnp.where(probs > math.e ** (-10), probs, math.e ** (-10))

    # normalize everything to sum to one
    return probs / probs.sum()


get_marginals_of_components_for_week_vectorized = vmap(get_marginals_of_components_for_week, in_axes=(0, 0, None))


class MixtureOfProducts(eqx.Module):
    centers: jax.Array
    scales: jax.Array
    weights: jax.Array
    n: int
    T: int
    coords: list  # we don't want to compute gradients w.r.t these parameters!

    def __init__(self, key, n, T, coords, scales, centers, weights):
        self.scales = scales
        self.centers = centers + 1e-4  # avoid nans in the gradient (which happened when grid cell centers were integer coordinates and could have distance 0 from grid cells)
        self.weights = weights
        self.n = n  # number of components
        self.T = T  # number of timesteps
        self.coords = coords

    """
    List[t: 1 - T][k: 1 - n] of jnp.array(cells[t]) -> jnp.array(cells[t])
    """
    def single_tstep_marginal(self, mu_k, weights):
        return (mu_k * weights.T).sum(axis=0)  # check this is correct

    """
    List[t: 1 - T][k: 1 - n] of jnp.array(cells[t]) -> jnp.array((cells[t], cells[t+1]))
    """
    def pairwise_marginal(self, mu_k1, mu_k2, weights):
        return (mu_k1 * weights.T).T.dot(mu_k2)

    def __call__(self):
        weights = jnp.expand_dims(softmax(self.weights), axis=0)
        test = jnp.zeros((self.n))
        single_tstep_marginals = []
        pairwise_marginals = []

        first = get_marginals_of_components_for_week_vectorized(self.centers[0], self.scales[0],
                                                                jnp.array(self.coords[0]))
        for t in range(1, self.T):
            second = get_marginals_of_components_for_week_vectorized(self.centers[t], self.scales[t],
                                                                     jnp.array(self.coords[t]))
            single_tstep_marginals.append(self.single_tstep_marginal(first, weights))
            pairwise_marginals.append(self.pairwise_marginal(first, second, weights))
            first = second
        single_tstep_marginals.append(
            self.single_tstep_marginal(first, weights))  # compute single tstep marginal for week T

        return single_tstep_marginals, pairwise_marginals