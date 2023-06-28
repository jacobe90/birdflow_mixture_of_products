from mixture_of_products import mixture_of_products_params, compute_marginal

a = mixture_of_products_params(2, 3, [4, 5, 6])
m = compute_marginal(a, [1, 2])
print(m)