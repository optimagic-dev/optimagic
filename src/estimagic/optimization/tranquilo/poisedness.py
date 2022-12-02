import numpy as np


def get_lagrange_poly_matrix(sample):
    """Construct matrix of lagrange polynomials.

    See :cite:`Conn2009`, Chapter 4.2, p. 60.

    Args:
        sample (np.ndarry): Array of shape (n_samples, n_params).

    Returns:
        np.ndarray: Matrix of lagrange polynomials of shape
            (n_samples, n_params * (n_params + 1) // 2).

    """
    basis_mat = _scaled_polynomial_features(sample)
    lagrange_mat = basis_mat @ np.linalg.inv(basis_mat.T @ basis_mat)

    return lagrange_mat


def _scaled_polynomial_features(x):
    """Construct linear terms, interactions and scaled square terms.

    The square terms are scaled by 1 / 2.

    Args:
        x (np.ndarray): Array of shape (n_samples, n_params).

    Returns:
        np.ndarray: Linear terms, interactions and scaled square terms.
            Has shape (n_samples, n_params * (n_params + 1) // 2).

    """
    n_samples, n_params = np.atleast_2d(x).shape
    n_poly_terms = _n_second_order_terms(n_params)

    poly_terms = np.empty((n_poly_terms, n_samples), np.float64)
    xt = x.T

    idx = 0
    for i in range(n_params):
        poly_terms[idx] = 0.5 * xt[i] ** 2
        idx += 1

        for j in range(i + 1, n_params):
            poly_terms[idx] = xt[i] * xt[j]
            idx += 1

    intercept = np.ones((1, n_samples), x.dtype)
    out = np.concatenate((intercept, xt, poly_terms), axis=0)

    return out.T


def _n_second_order_terms(dim):
    """Number of free second order terms in a quadratic model."""
    return dim * (dim + 1) // 2


def _reshape_coef_to_square_terms(coef, n_params):
    """Reshape coefficients to matrix of square terms."""
    mat = np.empty((n_params, n_params))
    idx = -1

    for j in range(n_params):
        for i in range(j + 1):
            idx += 1
            mat[i, j] = coef[idx]
            mat[j, i] = coef[idx]

    return mat


def _evaluate_scalar_model(x, intercept, linear_terms, square_terms):
    return intercept + linear_terms.T @ x + 0.5 * x.T @ square_terms @ x
