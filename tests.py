import unittest
from mixture_of_products_toy_implementation import compute_marginal, sample_route, mixture_of_products_params, \
    sample_locations_conditional, forecast
import numpy as np
from mixture_of_products_model import model_forward
from mixture_of_products_validation import track_log_likelihood, to_dynamic_conversion_dict
import mixture_of_products_model
from mixture_of_products_model_training import Datatuple, mask_input, pad_input
import haiku as hk
import jax.numpy as jnp
from jax import jit
import os
import h5py
import pandas as pd
import math
from functools import partial
from mixture_of_products_gaussian import MixtureOfProducts
import jax

def convert_toy_to_real(params):
    real_params = {}
    real_params['MixtureOfProductsModel'] = {'weights': jnp.asarray(params['weights'])}
    for t in range(len(params['locations'])):
        real_params['MixtureOfProductsModel'][f'week_{t}'] = jnp.asarray([params['products'][k][t] for k in range(params['n'])])
    return real_params

def get_single_tstep_marginal_gaussian(t, coords, scales, centers, weights):
    marginal = 0
    for k in range(len(weights)):
        probs = (1/(2*math.pi*((scales[t][k])**2)))*jnp.exp(-(0.5/((scales[t][k])**2))*(jnp.linalg.norm(coords[t] - centers[t][k], axis=1)**2))
        probs = jnp.where(probs > math.e ** (-10), probs, math.e ** (-10))
        marginal += jax.nn.softmax(weights)[k] * (probs / probs.sum())
    return marginal

def get_pairwise_marginal_gaussian(t, coords, scales, centers, weights):
    marginal = 0
    for k in range(len(weights)):
        probs_0 = (1/(2*math.pi*((scales[t][k])**2)))*jnp.exp(-(0.5/((scales[t][k])**2))*(jnp.linalg.norm(coords[t] - centers[t][k], axis=1)**2))
        probs_0 = jnp.where(probs_0 > math.e ** (-10), probs_0, math.e ** (-10))
        probs_0 /= probs_0.sum()
        probs_1 = (1 / (2 * math.pi * (scales[t+1][k]**2))) * jnp.exp(
            -(0.5 / (scales[t+1][k]**2)) * (jnp.linalg.norm(coords[t+1] - centers[t+1][k], axis=1) ** 2))
        probs_1 = jnp.where(probs_1 > math.e ** (-10), probs_1, math.e ** (-10))
        probs_1 /= probs_1.sum()
        marginal += jax.nn.softmax(weights)[k] * jnp.outer(probs_0, probs_1)
    return marginal
class TestEquinoxMopGaussianParameterization(unittest.TestCase):
    def test_single_timestep_marginals(self):
        coords = jnp.array([[[1, 1], [1, 0], [1, 2]], [[1, 1], [1, 0], [1, 2]], [[1, 1], [1, 0], [1, 2]], [[1, 1], [1, 0], [1, 2]]])
        key = jax.random.PRNGKey(42)
        model = MixtureOfProducts(key, n=4, T=4, coords=coords, scales=2*jnp.ones((4, 4)), centers=jnp.ones((4, 4, 2)), weights=jnp.ones(4))
        single, pairwise = model()
        for t in range(4):
            assert(jnp.allclose(single[t], get_single_tstep_marginal_gaussian(t, coords, model.scales, model.centers, model.weights)))
        for t in range(3):
            assert (jnp.allclose(pairwise[t], get_pairwise_marginal_gaussian(t, coords, model.scales, model.centers,
                                                                               model.weights)))
class TestValidation(unittest.TestCase):
    def test_track_log_likelihood_runs(self):
        hdf_src = "/Users/jacobepstein/Documents/work/BirdFlowModels/amewoo_2021_48km.hdf5"
        dist_pow = 0.4
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
        # amewoo_track_df = pd.concat([pd.read_csv(
        #     f"/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_black_box/tracks/amewoo-Blomberg-track-data-res-{res}km.csv"),
        #                              pd.read_csv(
        #                                  f"/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow_black_box/tracks/amewoo-track-data-res-{res}km.csv")],
        #                             axis=0, ignore_index=True)
        amewoo_track_df = pd.concat([pd.read_csv(
            f"/Users/jacobepstein/Documents/work/birdflow_black_box/tracks/amewoo-Blomberg-track-data-res-48km.csv"),
            pd.read_csv(
                f"/Users/jacobepstein/Documents/work/birdflow_black_box/tracks/amewoo-track-data-res-48km.csv")],
            axis=0, ignore_index=True)
        tll = track_log_likelihood(params, amewoo_track_df, masks)
        print(tll)
    def test_track_log_likelihood_is_correct(self):
        toy_params = {'n': 3, 'locations': [2, 2, 3, 3], 'weights': np.log([0.2, 0.2, 0.6]), 'products': [
            [np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])],
            [np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])],
            [np.log([0.2, 0.8]), np.log([0.5, 0.5]), np.log([0.1, 0.1, 0.8]), np.log([0.1, 0.2, 0.7])]]}
        params = convert_toy_to_real(toy_params)
        masks = [[True, False, True, False], [False, True, True, False], [True, True, False, True], [False, True, True, True]]
        conversion_dict = to_dynamic_conversion_dict(masks)
        track_df = {'cell.1': [2, 2], 'cell.2': [2, 3], 'st_week.1': [0, 1], 'st_week.2': [1, 2]}
        track_df = pd.DataFrame(data=track_df)
        tll_1 = track_log_likelihood(params, track_df, conversion_dict)
        tll_2 = 0.5 * (math.log(mixture_of_products_model.get_forecast_prob(params, [(1, 1)], [(0, 1)])) + math.log(mixture_of_products_model.get_forecast_prob(params, [(2, 2)], [(1, 1)])))
        tll_3 = 0.5 * (math.log(forecast(toy_params, [1], [(0, 1)])[1]) + math.log(forecast(toy_params, [2], [(1, 1)])[2]))
        print(tll_1, tll_2, tll_3)
        self.assertTrue(np.allclose(tll_1, tll_2))
        self.assertTrue(np.allclose(tll_1, tll_3))

    def test_track_log_likelihood_is_correct_ex_2(self):
        toy_params = {'n': 4, 'T': 5, 'locations': [2, 2, 2, 2, 2], 'weights': [0.1, 0.2, 0.3, 0.4],
                      'products': np.random.rand(4, 5, 2)}
        params = convert_toy_to_real(toy_params)
        track_df = {'cell.1': [4, 3, 3, 4], 'cell.2': [3, 3, 4, 4], 'st_week.1': [0, 1, 2, 3], 'st_week.2': [1, 2, 3, 4]}
        track_df = pd.DataFrame(data=track_df)
        masks = [[False, False, False, True, True],[False, False, False, True, True],[False, False, False, True, True],[False, False, False, True, True],[False, False, False, True, True]]
        conversion_dict = to_dynamic_conversion_dict(masks)
        print(conversion_dict)
        tll_1 = track_log_likelihood(params, track_df, conversion_dict)
        tll_2 = 0
        for i in range(len(track_df)):
            observations = [(int(track_df['st_week.2'][i]), conversion_dict[int(track_df['st_week.2'][i])][int(track_df['cell.2'][i])])]
            conditions = [(int(track_df['st_week.1'][i]), conversion_dict[int(track_df['st_week.1'][i])][int(track_df['cell.1'][i])])]
            tll_2 += 0.25 * math.log(mixture_of_products_model.get_forecast_prob(params, observations, conditions))
        print(tll_1, tll_2)
        forecast_probs = [forecast(toy_params, [1], [(0, 1)])[0], forecast(toy_params, [2], [(1, 0)])[0], forecast(toy_params, [3], [(2, 0)])[1], forecast(toy_params, [4], [(3, 1)])[1]]
        tll_3 = 0.25 * sum(list(map(math.log, forecast_probs)))
        print(tll_3)
        self.assertTrue(np.allclose(tll_1, tll_2))
        self.assertTrue(np.allclose(tll_1, tll_3))

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
        # print(params)
        # print(mixture_of_products_model.compute_marginal(params, [0]))
        # print(compute_marginal(toy_params, [0]))
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
    def test_model_forward(self):
        key = hk.PRNGSequence(42)
        params = model_forward.init(next(key), jnp.array([10, 10, 10]), 3, 10, learn_weights=True)
        weekly, pairwise = model_forward.apply(params, None, jnp.array([10, 10, 10]), 3, 10)
        print(weekly[0])
    def test_jit_compiled_init(self):
        key = hk.PRNGSequence(42)
        params = jit(partial(model_forward.init, cells=[1, 2, 3], rng=next(key), weeks=3, n=6, learn_weights=True,))()
        weekly, pairwise = model_forward.apply(params, None, jnp.array([1, 2, 3]), 3, 6)
        print(weekly)
        print(pairwise)
    def test_model_internal_compute_marginals_ex_1(self):
        key = hk.PRNGSequence(42)
        toy_params = {'n': 2, 'locations': [3, 3, 3], 'weights': [1, 1], 'products': np.log([[[0.2, 0.2, 0.6], [0.1, 0.5, 0.4], [0.6, 0.1, 0.3]], [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.4, 0.2, 0.4]]])}
        params = convert_toy_to_real(toy_params)
        weekly, pairwise = model_forward.apply(params, None, [3, 3, 3], 3, 2)
        print(weekly)
        print(pairwise)
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
        toy_weekly = [compute_marginal(toy_params, [t]) for t in range(len(toy_params['locations']))]
        toy_pairwise = [compute_marginal(toy_params, [t, t+1]) for t in range(len(toy_params['locations'])-1)]
        #toy_pairwise, toy_weekly = pad_input(toy_pairwise, toy_weekly, toy_params['locations'])
        print(f"weekly marginals: {weekly}")
        print(f"pairwise marginals: {pairwise}")
        print(f"")
        for i, marginal in enumerate(weekly):
            self.assertTrue(jnp.allclose(marginal, toy_weekly[i]))
        for i, marginal in enumerate(pairwise):
            self.assertTrue(jnp.allclose(marginal, toy_pairwise[i]))


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
