import numpy as np
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
    filtered_indices = np.ones(n_samples, dtype=np.int64) * -999
    filtered_indices[: n_params + 1] = indices[: n_params + 1]

    radius = state.radius
    center = state.x
    center_index = indices[state.index]
    centered_xs = (xs - center) / radius

    # refactor
    m_mat = np.zeros((n_samples, n_params + 1))
    m_mat[:, 0] = 1
    m_mat_pad = np.zeros((n_samples, n_samples))
    m_mat_pad[:n_samples, : n_params + 1] = m_mat

    n_mat = np.zeros((n_samples, n_poly_terms))

    m_mat[0, 1:] = centered_xs[indices[0]]
    n_mat[0, :] = _get_monomial_basis(m_mat[0, 1:])

    temp = 0
    for i in range(n_params + 1):
        if i == center_index:
            continue

        m_mat[i, 1:] = centered_xs[indices[i]]
        n_mat[i, :] = _get_monomial_basis(m_mat[i, 1:])
        temp += 1

        if temp == n_params:
            break

    counter = i + 1

    index = n_samples - 1 - counter

    while (counter < n_samples) and (index >= 0):

        if index == center_index and index != 0:
            continue

        m_mat[counter, 1:] = centered_xs[index]
        n_mat[counter, :] = _get_monomial_basis(m_mat[counter, 1:])

        m_mat_pad = np.zeros((n_samples, n_samples))
        m_mat_pad[:n_samples, : n_params + 1] = m_mat

        n_z_mat, _ = qr_multiply(
            m_mat_pad[: counter + 1, :],
            n_mat.T[:n_poly_terms, : counter + 1],
        )
        beta = np.linalg.svd(n_z_mat.T[n_params + 1 :], compute_uv=False)

        if beta[min(counter - n_params, n_poly_terms) - 1] > theta2:
            filtered_indices[counter] = index
            counter += 1

        index -= 1

    filtered_indices = filtered_indices[filtered_indices >= 0]

    return xs[filtered_indices], filtered_indices


def _get_monomial_basis(x):
    """Get the monomial basis (basis for quadratic functions) of x.

    Monomial basis = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n_params) ...
        ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n_params)^2]

    Args:
        x (np.ndarray): Parameter vector of shape (n_params,).

    Returns:
        np.ndarray: Monomial basis of x of shape (n_params * (n_params + 1) / 2,).
    """
    n_params = len(x)
    monomial_basis = np.zeros(int(n_params * (n_params + 1) / 2))

    j = 0
    for i in range(n_params):
        monomial_basis[j] = 0.5 * x[i] ** 2
        j += 1

        for k in range(i + 1, n_params):
            monomial_basis[j] = x[i] * x[k] / np.sqrt(2)
            j += 1

    return monomial_basis
