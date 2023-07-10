import unittest
from mixture_of_products_toy_implementation import compute_marginal, sample_route, mixture_of_products_params, \
    sample_locations_conditional, forecast
import numpy as np
from mixture_of_products_model import model_forward
from mixture_of_products_validation import track_log_likelihood
import mixture_of_products_model
from mixture_of_products_model_training import Datatuple, mask_input
import haiku as hk
import jax.numpy as jnp
import os
import h5py
import pandas as pd

def convert_toy_to_real(params):
    real_params = {}
    real_params['MixtureOfProductsModel'] = {'weights': jnp.asarray(params['weights'])}
    for k in range(len(params['products'])):
        product_k = params['products'][k]
        real_params[f'MixtureOfProductsModel/Product{k}'] = {}
        for i, marginal in enumerate(product_k):
            real_params[f'MixtureOfProductsModel/Product{k}'][f'week_{i}'] = jnp.asarray(marginal)
    return real_params

class TestValidation(unittest.TestCase):
    def test_track_log_likelihood_works(self):
        root = "/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_models"
        species = "amewoo"
        res = "48"
        dist_pow = 0.4
        hdf_src = os.path.join(root, f'{species}_2021_{res}km.hdf5')
        file = h5py.File(hdf_src, 'r')

        true_densities = np.asarray(file['distr']).T

        weeks = true_densities.shape[0]
        total_cells = true_densities.shape[1]

        distance_vector = np.asarray(file['distances']) ** dist_pow
        distance_vector *= 1 / (100 ** dist_pow)
        masks = np.asarray(file['geom']['dynamic_mask']).T.astype(bool)

        dtuple = Datatuple(weeks, total_cells, distance_vector, masks)
        distance_matrices, masked_densities = mask_input(true_densities, dtuple)
        cells = [d.shape[0] for d in masked_densities]
        key = hk.PRNGSequence(42)
        params = model_forward.init(next(key), cells, weeks, 10)
        amewoo_track_df = pd.concat([pd.read_csv(
            f"/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_black_box/tracks/amewoo-Blomberg-track-data-res-{res}km.csv"),
                                     pd.read_csv(
                                         f"/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_black_box/tracks/amewoo-track-data-res-{res}km.csv")],
                                    axis=0, ignore_index=True)
        tll = track_log_likelihood(params, amewoo_track_df)

class TestSampling(unittest.TestCase):
    def test_sample_route_works(self):
        toy_params = {'n': 2, 'locations': [3, 3, 3], 'weights': [1, 1], 'products': np.log(
            [[[0.2, 0.2, 0.6], [0.1, 0.5, 0.4], [0.6, 0.1, 0.3]], [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.4, 0.2, 0.4]]])}
        params = convert_toy_to_real(toy_params)
        print(mixture_of_products_model.sample_route(params))
    
class TestComputeMarginals(unittest.TestCase):
    def test_compute_marginals_ex_1(self):
        toy_params = {'n': 2, 'locations': [3, 3, 3], 'weights': [1, 1], 'products': np.log(
            [[[0.2, 0.2, 0.6], [0.1, 0.5, 0.4], [0.6, 0.1, 0.3]], [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.4, 0.2, 0.4]]])}
        params = convert_toy_to_real(toy_params)
        for t in range(2):
            self.assertTrue(jnp.allclose(mixture_of_products_model.compute_marginal(params, [t]), compute_marginal(toy_params, [t])))
        for t_1 in range(2):
            for t_2 in range(t_1+1, 2):
                self.assertTrue(jnp.allclose(mixture_of_products_model.compute_marginal(params, [t_1, t_2]),
                                             compute_marginal(toy_params, [t_1, t_2])))
    def test_compute_marginals_ex_2(self):
        toy_params = {'n': 3, 'locations': [2, 2, 3, 3], 'weights': np.log([0.2, 0.2, 0.6]), 'products': [
            [np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])],
            [np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])],
            [np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])]]}
        params = convert_toy_to_real(toy_params)
        for t in range(2):
            self.assertTrue(jnp.allclose(mixture_of_products_model.compute_marginal(params, [t]),
                                         compute_marginal(toy_params, [t])))
        for t_1 in range(2):
            for t_2 in range(t_1 + 1, 2):
                self.assertTrue(jnp.allclose(mixture_of_products_model.compute_marginal(params, [t_1, t_2]),
                                             compute_marginal(toy_params, [t_1, t_2])))

    def test_forecasting_ex1(self):
        toy_params = {'n': 2, 'T': 3, 'locations': [3, 3, 3], 'weights': np.log([0.4, 0.6]),
                  'products': [[np.log([0.2, 0.2, 0.6]), np.log([0.3, 0.2, 0.5]), np.log([0.7, 0.1, 0.2])],
                               [np.log([0.1, 0.1, 0.8]), np.log([0.3, 0.3, 0.4]), np.log([0.4, 0.4, 0.2])]]}
        params = convert_toy_to_real(toy_params)
        self.assertTrue(np.allclose(mixture_of_products_model.forecast(params, [1], [(0, 0)]), np.asarray([0.3, 0.24285714, 0.45714286])))
        self.assertTrue(np.allclose(mixture_of_products_model.forecast(params, [1], [(0, 1)]), np.asarray([0.3, 0.24285714, 0.45714286])))
        self.assertTrue(np.allclose(mixture_of_products_model.forecast(params, [1], [(0, 2)]), np.asarray([0.3, 0.26666667, 0.43333333])))

    def test_forecasting_ex2(self):
        toy_params = {'n': 4, 'T': 5, 'locations': [10, 10, 10, 10, 10], 'weights': [0.1, 0.2, 0.3, 0.4], 'products': np.random.rand(4, 5, 10)}
        params = convert_toy_to_real(toy_params)
        for t in range(4):
            for x in range(10):
                self.assertTrue(np.allclose(forecast(toy_params, [t+1], [(t, x)]), mixture_of_products_model.forecast(params, [t+1], [(t, x)])))
    
    def test_compute_marginal_prob(self):
        toy_params = {'n': 2, 'locations': [3, 3, 3], 'weights': [1, 1], 'products': np.log(
            [[[0.2, 0.2, 0.6], [0.1, 0.5, 0.4], [0.6, 0.1, 0.3]], [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.4, 0.2, 0.4]]])}
        params = convert_toy_to_real(toy_params)
        for t_1 in range(2):
            for t_2 in range(t_1+1, 2):
                marginal = mixture_of_products_model.compute_marginal(params, [t_1, t_2])
                for i in range(3):
                    for j in range(3):
                        self.assertTrue(np.allclose(marginal[i][j], mixture_of_products_model.get_prob(params, [(t_1, i), (t_2, j)])))
    
    def test_compute_conditional_prob(self):
        toy_params = {'n': 2, 'locations': [3, 3, 3], 'weights': [1, 1], 'products': np.log(
            [[[0.2, 0.2, 0.6], [0.1, 0.5, 0.4], [0.6, 0.1, 0.3]], [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.4, 0.2, 0.4]]])}
        params = convert_toy_to_real(toy_params)
        for t_1 in range(3):
            for t_2 in range(t_1+1, 3):
                for i in range(3):
                    conditional = mixture_of_products_model.forecast(params, [t_2], [(t_1, i)])
                    for j in range(3):
                        self.assertTrue(np.allclose(conditional[j], mixture_of_products_model.get_forecast_prob(params, [(t_2, j)], [(t_1, i)])))

class TestModelInternalComputeMarginals(unittest.TestCase):
    def test_model_internal_compute_marginals_ex_1(self):
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

    def test_model_internal_compute_marginals_ex_2(self):
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

    def test_forecasting_ex1(self):
        params = {'n': 2, 'T': 3, 'locations': [3, 3, 3], 'weights': np.log([0.4, 0.6]), 'products': [[np.log([0.2, 0.2, 0.6]), np.log([0.3, 0.2, 0.5]), np.log([0.7, 0.1, 0.2])], [np.log([0.1, 0.1, 0.8]), np.log([0.3, 0.3, 0.4]), np.log([0.4, 0.4, 0.2])]]}
        self.assertTrue(np.allclose(forecast(params, [1], [(0, 0)]), np.asarray([0.3, 0.24285714, 0.45714286])))
        self.assertTrue(np.allclose(forecast(params, [1], [(0, 1)]), np.asarray([0.3, 0.24285714, 0.45714286])))
        self.assertTrue(np.allclose(forecast(params, [1], [(0, 2)]), np.asarray([0.3, 0.26666667, 0.43333333])))

    def test_forecasting_ex2(self):
        params = {'n': 4, 'T': 5, 'locations': [10, 10, 10, 10, 10], 'weights': [0.1, 0.2, 0.3, 0.4], 'products': np.random.rand(4, 5, 10)}

        def forecast_unvectorized(params, tsteps, observations):
            all_tsteps = list(map(lambda tup: tup[0], observations)) + list(tsteps)
            joint = compute_marginal(params, all_tsteps)
            conditional = joint
            for (t, obs) in observations:
                conditional = conditional[obs]

            observation_marginal = compute_marginal(params, list(map(lambda tup: tup[0], observations)))
            scale_factor = observation_marginal
            for (t, obs) in observations:
                scale_factor = scale_factor[obs]
            conditional /= scale_factor
            return conditional

        print(forecast_unvectorized(params, [4], [(3, 2)]))
        for t in range(4):
            for x in range(10):
                self.assertTrue(np.allclose(forecast(params, [t+1], [(t, x)]), forecast_unvectorized(params, [t+1], [(t, x)])))

    def test_forecasting(self):
        params = mixture_of_products_params(10, 7, [20] * 7)
        conditional = forecast(params, [1, 2], [(0, 10)])
        print(conditional)


if __name__ == "__main__":
    unittest.main()