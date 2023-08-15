# box radius 1 experiment
sbatch train_mixture_of_products_from_sampled_routes.sh -r 2000 -d 1

# box radius 2 experiments
sbatch train_mixture_of_products_from_sampled_routes.sh -r 10 -d 2
sbatch train_mixture_of_products_from_sampled_routes.sh -r 100 -d 2
sbatch train_mixture_of_products_from_sampled_routes.sh -r 150 -d 2
sbatch train_mixture_of_products_from_sampled_routes.sh -r 250 -d 2
sbatch train_mixture_of_products_from_sampled_routes.sh -r 450 -d 2
sbatch train_mixture_of_products_from_sampled_routes.sh -r 1000 -d 2
sbatch train_mixture_of_products_from_sampled_routes.sh -r 2000 -d 2