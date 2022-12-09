from functools import partial

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


def get_poisedness_constant(sample):
    """Calculate the lambda poisedness constant.

    The implementation is based on :cite:`Conn2009`, Chapters 3 and 4.

    Args:
        sample (np.ndarry): Array of shape (n_samples, n_params).

    Returns:
        float: The lambda poisedness constant.

    """
    n_params = sample.shape[1]

    lagrange_mat = lagrange_poly_matrix(sample)
    center = np.zeros(n_params)

    lambda_ = 0
    for poly in lagrange_mat:

        intercept = poly[0]
        linear_terms = poly[1 : n_params + 1]
        _coef_square_terms = poly[n_params + 1 :]
        square_terms = _reshape_coef_to_square_terms(_coef_square_terms, n_params)

        nonlinear_constraint = NonlinearConstraint(lambda x: np.linalg.norm(x), 0, 1)
        _func_to_maximize = partial(
            _get_neg_absolute_value,
            intercept=intercept,
            linear_terms=linear_terms,
            square_terms=square_terms,
        )
        results_max = minimize(
            _func_to_maximize,
            center,
            method="trust-constr",
            constraints=[nonlinear_constraint],
        )
        critval = _get_absolute_value(
            results_max.x, intercept, linear_terms, square_terms
        )

        if critval > lambda_:
            lambda_ = critval

    return lambda_


def lagrange_poly_matrix(sample):
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
    n_poly_terms = n_params * (n_params + 1) // 2

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


def _reshape_coef_to_square_terms(coef, n_params):
    """Reshape square coefficients to matrix of square terms."""
    mat = np.empty((n_params, n_params))
    idx = -1

    for j in range(n_params):
        for i in range(j + 1):
            idx += 1
            mat[i, j] = coef[idx]
            mat[j, i] = coef[idx]

    return mat


def _get_absolute_value(x, intercept, linear_terms, square_terms):
    return np.abs(intercept + linear_terms.T @ x + 0.5 * x.T @ square_terms @ x)


def _get_neg_absolute_value(x, intercept, linear_terms, square_terms):
    return -_get_absolute_value(x, intercept, linear_terms, square_terms)
