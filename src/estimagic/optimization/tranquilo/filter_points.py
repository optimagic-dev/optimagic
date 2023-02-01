import numpy as np
from numba import njit
from scipy.linalg import qr_multiply

from estimagic.optimization.tranquilo.clustering import cluster
from estimagic.optimization.tranquilo.get_component import get_component
from estimagic.optimization.tranquilo.models import n_second_order_terms
from estimagic.optimization.tranquilo.volume import get_radius_after_volume_scaling


def get_sample_filter(sample_filter="keep_all", user_options=None):
    """Get filter function with partialled options.

    The filter function is applied to points inside the current trustregion before
    additional points are sampled.

    The resulting function only takes an array of shape n_points, n_params as argument.

    Args:
        filter (str or callable): The name of a built in filter or a function with the
            filter interface.

    Returns:
        callable: The filter

    """
    built_in_filters = {
        "discard_all": discard_all,
        "keep_all": keep_all,
        "clustering": keep_cluster_centers,
        "keep_sphere": keep_sphere,
        "drop_pounders": drop_collinear_pounders,
    }

    out = get_component(
        name_or_func=sample_filter,
        component_name="sample_filter",
        func_dict=built_in_filters,
        user_options=user_options,
    )

    return out


def discard_all(state):
    return state.x.reshape(1, -1), np.array([state.index])


def keep_all(xs, indices):
    return xs, indices


def keep_sphere(xs, indices, state):
    dists = np.linalg.norm(xs - state.trustregion.center, axis=1)
    keep = dists <= state.trustregion.radius
    return xs[keep], indices[keep]


def drop_collinear_pounders(xs, indices, state):
    """Drop collinear points using pounders filtering."""
    if xs.shape[0] <= xs.shape[1] + 1:
        filtered_xs, filtered_indices = xs, indices
    else:
        filtered_xs, filtered_indices = _drop_collinear_pounders(xs, indices, state)

    return filtered_xs, filtered_indices


def keep_cluster_centers(
    xs, indices, state, target_size, strictness=1e-10, shape="sphere"
):
    dim = xs.shape[1]
    scaling_factor = strictness / target_size
    cluster_radius = get_radius_after_volume_scaling(
        radius=state.trustregion.radius,
        dim=dim,
        scaling_factor=scaling_factor,
    )
    _, centers = cluster(x=xs, epsilon=cluster_radius, shape=shape)

    # do I need to make sure trustregion center is in there?
    out = xs[centers], indices[centers]
    return out


def _drop_collinear_pounders(xs, indices, state):
    theta2 = 1e-4
    n_samples, n_params = xs.shape
    n_poly_terms = n_second_order_terms(n_params)

    indices_reverse = indices[::-1]
    indexer_reverse = np.arange(n_samples)[::-1]

    radius = state.trustregion.radius
    center = state.trustregion.center
    index_center = int(np.where(indices_reverse == state.index)[0])
    centered_xs = (xs - center) / radius

    (
        linear_features,
        square_features,
        idx_list_n_plus_1,
        index,
    ) = _get_polynomial_feature_matrices(
        centered_xs,
        indexer_reverse,
        index_center,
        n_params,
        n_samples,
        n_poly_terms,
    )

    indexer_filtered = indexer_reverse[idx_list_n_plus_1].tolist()
    _index_center = indexer_reverse[index_center]

    counter = n_params + 1

    while (counter < n_samples) and (index >= 0):
        if index == _index_center:
            index -= 1
            continue

        linear_features[counter, 1:] = centered_xs[index]
        square_features[counter, :] = _scaled_square_features(
            linear_features[counter, 1:]
        )

        linear_features_pad = np.zeros((n_samples, n_samples))
        linear_features_pad[:n_samples, : n_params + 1] = linear_features

        n_z_mat, _ = qr_multiply(
            linear_features_pad[: counter + 1, :],
            square_features.T[:n_poly_terms, : counter + 1],
        )
        beta = np.linalg.svd(n_z_mat.T[n_params + 1 :], compute_uv=False)

        if beta[min(counter - n_params, n_poly_terms) - 1] > theta2:
            indexer_filtered += [index]
            counter += 1

        index -= 1

    filtered_indices = indices[indexer_filtered]
    filtered_xs = xs[indexer_filtered]

    return filtered_xs, filtered_indices


def _get_polynomial_feature_matrices(
    centered_xs, indexer, index_center, n_params, n_samples, n_poly_terms
):
    linear_features = np.zeros((n_samples, n_params + 1))
    square_features = np.zeros((n_samples, n_poly_terms))

    linear_features[0, 1:] = centered_xs[indexer[index_center]]
    square_features[0, :] = _scaled_square_features(linear_features[0, 1:]).flatten()

    _is_center_in_head = index_center < n_params
    idx_list_n = [i for i in range(n_params + _is_center_in_head) if i != index_center]
    idx_list_n_plus_1 = [index_center, *idx_list_n]

    linear_features[:, 0] = 1
    linear_features[: n_params + 1, 1:] = centered_xs[indexer[idx_list_n_plus_1]]
    square_features[: n_params + 1, :] = _scaled_square_features(
        linear_features[: n_params + 1, 1:]
    )

    idx = n_samples - _is_center_in_head - len(idx_list_n) - 1

    return linear_features, square_features, idx_list_n_plus_1, idx


@njit
def _scaled_square_features(x):
    """Construct scaled interaction and square terms.

    The interaction terms are scaled by 1 / sqrt{2} while the square terms are scaled
    by 1 / 2.

    Args:
        x (np.ndarray): Array of shape (n_samples, n_params).

    Returns:
        np.ndarray: Scaled interaction and square terms. Has shape (n_samples,
            n_params + (n_params - 1) * n_params / 1).

    """
    n_samples, n_params = np.atleast_2d(x).shape
    n_poly_terms = n_second_order_terms(n_params)

    poly_terms = np.empty((n_poly_terms, n_samples), np.float64)
    xt = x.T

    idx = 0
    for i in range(n_params):
        poly_terms[idx] = xt[i] ** 2 / 2
        idx += 1

        for j in range(i + 1, n_params):
            poly_terms[idx] = xt[i] * xt[j] / np.sqrt(2)
            idx += 1

    return poly_terms.T
