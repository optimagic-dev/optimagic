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
        "discard_all": _discard_all,
        "keep_all": _keep_all,
        "drop_collinear": _drop_collinear,
        "drop_pounders": _drop_collinear_pounders,
    }

    if isinstance(sample_filter, str) and sample_filter in built_in_filters:
        out = built_in_filters[sample_filter]
    elif callable(sample_filter):
        out = sample_filter
    else:
        raise ValueError()

    return out


def _discard_all(xs, indices, state):
    return state.x.reshape(1, -1), np.array([state.index])


def _keep_all(xs, indices, state):
    return xs, indices


def _drop_collinear(xs, indices, state):
    """Make sure that the points that are kept are linearly independent."""
    raise NotImplementedError()


def _drop_collinear_pounders(xs, indices, state):
    """Drop collinear points using pounders filtering.


    Args:
        xs (np.ndarray): Old xs to filter
        indices (np.ndarray): Old indices from history to filter
        state (class): Class containing current best/accepted x: state.x


    Returns:
        tuple:
        - (np.ndarray): Filtered xs.
        - (np.ndarray): Filtered indices.

    """
    theta2 = 1e-4
    n_samples, n_params = xs.shape
    n_poly_terms = n_params * (n_params + 1) // 2

    indices = np.flip(indices)

    radius = state.radius
    center = state.x
    center_index = indices[state.index]
    centered_xs = (xs - center) / radius

    (
        feature_mat_linear,
        feature_mat_square,
        _idx_list,
        index,
    ) = _get_polynomial_feature_matrices(
        centered_xs, indices, center_index, n_params, n_samples, n_poly_terms
    )

    filtered_indices = np.ones(n_samples, dtype=np.int64) * -999
    filtered_indices[0] = indices[center_index]
    filtered_indices[1 : n_params + 1] = indices[_idx_list]

    counter = n_params + 1

    while (counter < n_samples) and (index >= 0):

        if index == indices[center_index]:
            index -= 1
            continue

        feature_mat_linear[counter, 1:] = centered_xs[index]
        feature_mat_square[counter, :] = _get_monomial_basis(
            feature_mat_linear[counter, 1:]
        )

        m_mat_pad = np.zeros((n_samples, n_samples))
        m_mat_pad[:n_samples, : n_params + 1] = feature_mat_linear

        n_z_mat, _ = qr_multiply(
            m_mat_pad[: counter + 1, :],
            feature_mat_square.T[:n_poly_terms, : counter + 1],
        )
        beta = np.linalg.svd(n_z_mat.T[n_params + 1 :], compute_uv=False)

        if beta[min(counter - n_params, n_poly_terms) - 1] > theta2:
            filtered_indices[counter] = index
            counter += 1

        index -= 1

    filtered_indices = filtered_indices[filtered_indices >= 0]

    return xs[filtered_indices], filtered_indices


def _get_polynomial_feature_matrices(
    centered_xs, indices, center_index, n_params, n_samples, n_poly_terms
):
    m_mat = np.zeros((n_samples, n_params + 1))
    n_mat = np.zeros((n_samples, n_poly_terms))

    m_mat[0, 1:] = centered_xs[indices[0]]
    n_mat[0, :] = _get_monomial_basis(m_mat[0, 1:]).flatten()

    idx_list = [center_index]
    _is_center_in_head = center_index < n_params
    _idx_list = [i for i in range(n_params + _is_center_in_head) if i != center_index]
    idx_list = [center_index] + _idx_list

    m_mat[:, 0] = 1
    m_mat[: n_params + 1, 1:] = centered_xs[indices[idx_list]]
    n_mat[: n_params + 1, :] = _get_monomial_basis(m_mat[: n_params + 1, 1:])

    index = n_samples - _is_center_in_head - len(_idx_list) - 1

    return m_mat, n_mat, _idx_list, index


@njit
def _get_monomial_basis(x):
    """Get the monomial basis (basis for quadratic functions) of x.

    Monomial basis = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n_params) ...
        ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n_params)^2]

    Args:
        x (np.ndarray): Parameter vector of shape (n_params,).

    Returns:
        np.ndarray: Monomial basis of x of shape (n_params * (n_params + 1) / 2,).
    """
    n_samples, n_params = np.atleast_2d(x).shape
    has_squares = True

    if has_squares:
        n_poly_terms = n_params * (n_params + 1) // 2
    else:
        n_poly_terms = n_params * (n_params - 1) // 2

    poly_terms = np.empty((n_poly_terms, n_samples), x.dtype)
    xt = x.T

    idx = 0
    for i in range(n_params):
        poly_terms[idx] = 0.5 * xt[i] ** 2
        idx += 1

        # start i + 1 because has squares?
        for j in range(i + 1, n_params):
            poly_terms[idx] = xt[i] * xt[j] / np.sqrt(2)
            idx += 1

    return poly_terms.T


@njit
def _polynomial_features(x, has_intercepts, has_squares):
    n_samples, n_params = x.shape

    if has_squares:
        n_poly_terms = n_params * (n_params + 1) // 2
    else:
        n_poly_terms = n_params * (n_params - 1) // 2

    poly_terms = np.empty((n_poly_terms, n_samples), x.dtype)
    xt = x.T

    idx = 0
    for i in range(n_params):
        j_start = i if has_squares else i + 1
        for j in range(j_start, n_params):
            poly_terms[idx] = xt[i] * xt[j]
            idx += 1

    if has_intercepts:
        intercept = np.ones((1, n_samples), x.dtype)
        out = np.concatenate((intercept, xt, poly_terms), axis=0)
    else:
        out = np.concatenate((xt, poly_terms), axis=0)

    return out.T
