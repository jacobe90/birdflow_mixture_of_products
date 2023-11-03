import jax.numpy as jnp
import equinox as eqx
from mixture_of_products_gaussian import MixtureOfProducts

"""
Functions for generating coordinate conversion array
coords is an array of length T, coords[t][i] = coordinates of weekly mask t cell i in the overall grid
"""
def generate_coords_array(cells, masks, nan_mask, x_dim, y_dim):
    coords = []
    for t in range(len(cells)):
        coords.append([])
        for i in range(cells[t]):
            cell = get_index_in_bigger_grid(get_index_in_bigger_grid(i, masks[t]), nan_mask)
            coords[t].append(cell_to_xy(cell, x_dim, y_dim))
    return coords

def get_index_in_bigger_grid(cell, mask):
    true_count = -1
    new_cell = None
    for i, b in enumerate(mask):
        if b:
            true_count += 1
        if true_count == cell:
            new_cell = i
            break
    return new_cell

def cell_to_xy(cell, x_dim, y_dim):
    x = cell % x_dim
    y = int(cell / x_dim)
    return [x, y]


"""
Forward pass / training loop
"""
def obs_loss(pred_densities, true_densities):
    obs = 0
    for pred, true in zip(pred_densities, true_densities):
        residual = true - pred
        obs += jnp.sum(jnp.square(residual))
    return obs


def distance_loss(flows, d_matrices):
    dist = 0
    for flow, d_matrix in zip(flows, d_matrices):
        dist += jnp.sum(flow * d_matrix)
    return dist


def entropy(probs):
    logp = jnp.log(probs)
    ent = probs * logp
    h = -1 * jnp.sum(ent)
    return h


def ent_loss(probs, flows):
    ent = 0
    for p in probs:
        ent += entropy(p)
    for f in flows:
        ent -= entropy(f)
    return ent


def loss_fn(model, true_densities, d_matrices, obs_weight, dist_weight, ent_weight):
    pred_densities, flows = model()
    obs = obs_loss(pred_densities, true_densities)
    dist = distance_loss(flows, d_matrices)
    ent = ent_loss(flows, pred_densities)

    return (obs_weight * obs) + (dist_weight * dist) + (-1 * ent_weight * ent), (obs, dist, ent)


def train_model(loss_fn,
                optimizer,
                training_steps,
                n,
                T,
                coords,
                scales,
                centers,
                weights,
                key):

    model = MixtureOfProducts(key, n, T, coords, scales, centers, weights)
    # initialize optimizer state, make sure we don't compute gradients of coords! (could this cause a problem?)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def update(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_model = eqx.apply_updates(model, updates) #eqx.apply_updates or optax.apply_updates?
        return new_model, new_opt_state, loss

    update = eqx.filter_jit(update)  # could be a problem because coords will get marked by jit

    loss_dict = {
        'total': [],
        'obs': [],
        'dist': [],
        'ent': [],
    }

    for step in range(training_steps):
        model, opt_state, loss = update(model, opt_state)
        total_loss, loss_components = loss
        obs, dist, ent = loss_components
        loss_dict['total'].append(float(total_loss))
        loss_dict['obs'].append(float(obs))
        loss_dict['dist'].append(float(dist))
        loss_dict['ent'].append(float(ent))

    return model, loss_dict