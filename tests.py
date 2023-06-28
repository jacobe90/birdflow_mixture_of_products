import unittest
from mixture_of_products import compute_marginal, softmax
import numpy as np

class TestComputeMarginals(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()