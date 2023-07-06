import haiku as hk
from jax.nn import softmax
import jax.numpy as jnp
from jax.random import categorical

class Product(hk.Module):
    def __init__(self, cells, idx):
        super().__init__(name=f"Product{idx}")
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
    def __init__(self, cells, weeks, n, name="MixtureOfProductsModel"):
        super().__init__(name=name)
        self.weeks = weeks
        self.cells = cells
        self.n = n
        self.products = []

    def get_marginal(self, weights, tsteps):
        marginal = 0
        for k in range(self.n):
            prod_k_marginal = jnp.asarray(1)
            for tstep in tsteps:
                prod_k_marginal = jnp.tensordot(prod_k_marginal, self.products[k](tstep), axes=0)
            marginal += weights[k] * prod_k_marginal
        return marginal

    def __call__(self):
        # initialize weights
        weights = hk.get_parameter(
            'weights',
            (self.n,),
            init=hk.initializers.RandomNormal(),
            dtype='float32'
        )
        weights = softmax(weights, axis=0)

        # initialize product distributions
        for k in range(self.n):
            self.products.append(Product(self.cells, k))

        #TODO: vectorize everything more somehow?
        single_tstep_marginals = []
        pairwise_marginals = []
        # for t in range(self.weeks):
        #     single_tstep_marginal = 0
        #     for k in range(self.n):
        #         prod_k_marginal = jnp.asarray(1)
        #         for tstep in [t]:
        #             prod_k_marginal = jnp.tensordot(prod_k_marginal, self.products[k](tstep), axes=0)
        #         single_tstep_marginal += weights[k] * prod_k_marginal
        #     single_tstep_marginals.append(single_tstep_marginal)
        #
        # for t in range(self.weeks - 1):
        #     pairwise_marginal = 0
        #     for k in range(self.n):
        #         prod_k_marginal = jnp.asarray(1)
        #         for tstep in [t, t + 1]:
        #             prod_k_marginal = jnp.tensordot(prod_k_marginal, self.products[k](tstep), axes=0)
        #         pairwise_marginal += weights[k] * prod_k_marginal
        #     pairwise_marginals.append(pairwise_marginal)
        single_tstep_marginals = [self.get_marginal(weights, [t]) for t in range(self.weeks)]
        pairwise_marginals = [self.get_marginal(weights, [t, t+1]) for t in range(self.weeks-1)]

        return single_tstep_marginals, pairwise_marginals


def predict(cells, weeks, n):
    model = MixtureOfProductsModel(cells, weeks, n)
    return model()


model_forward = hk.transform(predict)
