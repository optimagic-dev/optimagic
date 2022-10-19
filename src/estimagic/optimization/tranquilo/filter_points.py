import numpy as np
from numba import njit
from scipy.linalg import qr_multiply


def get_sample_filter(sample_filter="keep_all"):
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
        "drop_collinear": drop_collinear,
        "drop_pounders": drop_collinear_pounders,
    }

    if isinstance(sample_filter, str) and sample_filter in built_in_filters:
        out = built_in_filters[sample_filter]
    elif callable(sample_filter):
        out = sample_filter
    else:
        raise ValueError()

    return out


def discard_all(xs, indices, state):
    return state.x.reshape(1, -1), np.array([state.index])


def keep_all(xs, indices, state):
    return xs, indices


def drop_collinear(xs, indices, state):
    """Make sure that the points that are kept are linearly independent."""
    raise NotImplementedError()


def drop_collinear_pounders(xs, indices, state):
    """Drop collinear points using pounders filtering."""
    theta2 = 1e-4
    n_samples, n_params = xs.shape
    n_poly_terms = n_params * (n_params + 1) // 2

    indices = np.flip(indices)

    radius = state.radius
    center = state.x
    index_center = indices[state.index]
    centered_xs = (xs - center) / radius

    (
        linear_features,
        square_features,
        index_list_additional,
        index,
    ) = _get_polynomial_feature_matrices(
        centered_xs, indices, index_center, n_params, n_samples, n_poly_terms
    )

    filtered_indices = np.ones(n_samples, dtype=np.int64) * -999
    filtered_indices[0] = indices[index_center]
    filtered_indices[1 : n_params + 1] = indices[index_list_additional]

    counter = n_params + 1

    while (counter < n_samples) and (index >= 0):

        if index == indices[index_center]:
            index -= 1
            continue

        linear_features[counter, 1:] = centered_xs[index]
        square_features[counter, :] = _square_features(linear_features[counter, 1:])

        linear_features_pad = np.zeros((n_samples, n_samples))
        linear_features_pad[:n_samples, : n_params + 1] = linear_features

        n_z_mat, _ = qr_multiply(
            linear_features_pad[: counter + 1, :],
            square_features.T[:n_poly_terms, : counter + 1],
        )
        beta = np.linalg.svd(n_z_mat.T[n_params + 1 :], compute_uv=False)

        if beta[min(counter - n_params, n_poly_terms) - 1] > theta2:
            filtered_indices[counter] = index
            counter += 1

        index -= 1

    filtered_indices = filtered_indices[filtered_indices >= 0]

    return xs[filtered_indices], filtered_indices


def _get_polynomial_feature_matrices(
    centered_xs, indices, index_center, n_params, n_samples, n_poly_terms
):
    linear_features = np.zeros((n_samples, n_params + 1))
    square_features = np.zeros((n_samples, n_poly_terms))

    linear_features[0, 1:] = centered_xs[indices[0]]
    square_features[0, :] = _square_features(linear_features[0, 1:]).flatten()

    _is_center_in_head = index_center < n_params
    idx_list_n = [i for i in range(n_params + _is_center_in_head) if i != index_center]
    idx_list_n_plus_1 = [index_center] + idx_list_n

    linear_features[:, 0] = 1
    linear_features[: n_params + 1, 1:] = centered_xs[indices[idx_list_n_plus_1]]
    square_features[: n_params + 1, :] = _square_features(
        linear_features[: n_params + 1, 1:]
    )

    idx = n_samples - _is_center_in_head - len(idx_list_n) - 1

    return linear_features, square_features, idx_list_n, idx


@njit
def _square_features(x):
    n_samples, n_params = np.atleast_2d(x).shape
    n_poly_terms = n_params * (n_params + 1) // 2

    poly_terms = np.empty((n_poly_terms, n_samples), x.dtype)
    xt = x.T

    idx = 0
    for i in range(n_params):
        poly_terms[idx] = 0.5 * xt[i] ** 2
        idx += 1

        for j in range(i + 1, n_params):
            poly_terms[idx] = xt[i] * xt[j] / np.sqrt(2)
            idx += 1

    return poly_terms.T
