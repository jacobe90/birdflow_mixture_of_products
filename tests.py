import unittest
from mixture_of_products_toy_implementation import compute_marginal, sample_route, mixture_of_products_params, \
    sample_locations_conditional, forecast
import numpy as np
from mixture_of_products_model import model_forward
import haiku as hk
import jax.numpy as jnp

def convert_toy_to_real(params):
    real_params = {}
    real_params['MixtureOfProductsModel'] = {'weights': jnp.asarray(params['weights'])}
    for k in range(len(params['products'])):
        product_k = params['products'][k]
        real_params[f'MixtureOfProductsModel/Product{k}'] = {}
        for i, marginal in enumerate(product_k):
            real_params[f'MixtureOfProductsModel/Product{k}'][f'week_{i}'] = jnp.asarray(marginal)
    return real_params

class TestComputeMarginals(unittest.TestCase):
    def test_example_1(self):
        key = hk.PRNGSequence(42)
        params = {'MixtureOfProductsModel': {'weights': jnp.asarray([1, 1])}, 'MixtureOfProductsModel/Product0': {'week_0': jnp.log(jnp.asarray([0.2 , 0.2,  0.6])), 'week_1': jnp.log(jnp.asarray([0.1 , 0.5,  0.4])), 'week_2': jnp.log(jnp.asarray([0.6 , 0.1,  0.3]))}, 'MixtureOfProductsModel/Product1': {'week_0': jnp.log(jnp.asarray([0.1 , 0.1,  0.8])), 'week_1': jnp.log(jnp.asarray([0.7 , 0.2,  0.1])), 'week_2': jnp.log(jnp.asarray([0.4 , 0.2,  0.4]))}}
        toy_params = {'n': 2, 'locations': [3, 3, 3], 'weights': [1, 1], 'products': np.log([[[0.2, 0.2, 0.6], [0.1, 0.5, 0.4], [0.6, 0.1, 0.3]], [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.4, 0.2, 0.4]]])}
        weekly, pairwise = model_forward.apply(params, None, [3, 3, 3], 3, 2)

        for i, marginal in enumerate(weekly):
            self.assertTrue(jnp.allclose(marginal, compute_marginal(toy_params, [i])))
        for i, marginal in enumerate(pairwise):
            self.assertTrue(jnp.allclose(marginal, compute_marginal(toy_params, [i, i+1])))

        # check marginal computed by hand matches the model-computed marginal
        true_marginal = 0.5 * (jnp.asarray([[0.06, 0.01, 0.03], [0.3, 0.05, 0.15], [0.24, 0.04, 0.12]]) + jnp.asarray(
            [[0.28, 0.14, 0.28], [0.08, 0.04, 0.08], [0.04, 0.02, 0.04]]))
        self.assertTrue(jnp.allclose(pairwise[1], true_marginal))

    def test_example_2(self):
        toy_params = {'n': 3, 'locations': [2, 2, 3, 3], 'weights': np.log([0.2, 0.2, 0.6]), 'products': [[np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])],
                                                                                            [np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])],
                                                                                            [np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])]]}
        params = convert_toy_to_real(toy_params)
        weekly, pairwise = model_forward.apply(params, None, [2, 2, 3, 3], 4, 3)
        for i, marginal in enumerate(weekly):
            self.assertTrue(jnp.allclose(marginal, compute_marginal(toy_params, [i])))
        for i, marginal in enumerate(pairwise):
            self.assertTrue(jnp.allclose(marginal, compute_marginal(toy_params, [i, i+1])))


class TestToyComputeMarginals(unittest.TestCase):
    def test_toy_example_1(self):
        params = {'n': 2, 'locations': [3, 3, 3], 'weights': [1, 1], 'products': np.log([[[0.2, 0.2, 0.6], [0.1, 0.5, 0.4], [0.6, 0.1, 0.3]], [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.4, 0.2, 0.4]]])}
        marginal = compute_marginal(params, [1, 2])
        true_marginal = 0.5 * (np.asarray([[0.06, 0.01, 0.03], [0.3, 0.05, 0.15], [0.24, 0.04, 0.12]]) + np.asarray([[0.28, 0.14, 0.28], [0.08, 0.04, 0.08], [0.04, 0.02, 0.04]]))
        self.assertTrue(np.allclose(marginal, true_marginal))

    def test_toy_example_2(self):
        params = {'n': 3, 'locations': [2, 2, 3, 3], 'weights': np.log([0.2, 0.2, 0.6]), 'products': [[np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])],
                                                                                            [np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])],
                                                                                            [np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])]]}
        marginal_02 = compute_marginal(params, [0, 2])
        marginal_013 = compute_marginal(params, [0, 1, 3])
        true_02 = np.asarray([[0.02, 0.02, 0.16], [0.08, 0.08, 0.64]])
        true_013 = np.asarray([[[0.01, 0.02, 0.07], [0.01, 0.02, 0.07]],
                               [[0.04, 0.08, 0.28], [0.04, 0.08, 0.28]]])
        self.assertTrue(np.allclose(marginal_02, true_02))
        self.assertTrue(np.allclose(marginal_013, true_013))

    def test_sampling_works(self):
        params = mixture_of_products_params(10, 7, [100]*7)
        route = sample_route(params)
        conditional_route = sample_locations_conditional(params, [0, 2, 3], [(4, 99), (6, 2)])
        print(route, conditional_route)

    def test_forecasting(self):
        params = mixture_of_products_params(10, 7, [20] * 7)
        conditional = forecast(params, [1, 2], [(0, 10)])
        print(conditional)


if __name__ == "__main__":
    unittest.main()