from functools import partial

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize


def get_poisedness_constant(sample, shape="sphere"):
    """Calculate the lambda poisedness constant of a sample.

    Note that the sample space is a trust-region with center 0 and radius 1.
    It may be a (hyper-) sphere or cube.

    The implementation is based on :cite:`Conn2009`, Chapters 3 and 4.

    In general, if the sample is lambda-poised with a small lambda, where lambda >=1,
    the sample is said to have "good" geometry or "spans" the trust-region well.
    As lambda grows, the system represented by these points becomes increasingly
    linearly dependent.

    Formal definition:
    A sample Y is said to be lambda-poised on a region of interest if it is linearly
    independent and the Lagrange polynomials L(i) of points i through N in Y satisfy:

        lambda >= max_i max_x | L(i) |      (1)

    i.e. for each point i in the sample, we maximize the absolute criterion value
    of its lagrange polynomial L(i); we then take the maximum over all these
    criterion values as the lambda constant.

    When we compare different samples on the same trust-region, we are usually
    interested in keeping the sample with the least lambda, so that (1) holds.


    Args:
        sample (np.ndarry): Array of shape (n_samples, n_params) containing the scaled
            sample of points that lie within a trust-region with center 0 and radius 1.
        shape (str): Geometric shape of the sample space. One of "sphere", "cube".
            Default is "sphere".

    Returns:
        tuple:
            - lambda (float): The lambda poisedness constant.
            - argmax (np.ndarray): 1d array of shape (n_params,) containing the
                parameter vector that maximizes lambda.
            - idx_max (int): Index relating to the position of the argmax in the sample.

    """
    n_params = sample.shape[1]
    options = _get_minimize_options(shape, n_params)

    center = np.zeros(n_params)
    lagrange_mat = _lagrange_poly_matrix(sample)

    lambda_ = 0
    idx_max = None

    for idx, poly in enumerate(lagrange_mat):
        intercept = poly[0]
        linear_terms = poly[1 : n_params + 1]
        _coef_square_terms = poly[n_params + 1 :]
        square_terms = _reshape_coef_to_square_terms(_coef_square_terms, n_params)

        neg_criterion = partial(
            _eval_neg_absolute_value,
            intercept=intercept,
            linear_terms=linear_terms,
            square_terms=square_terms,
        )

        result_max = minimize(fun=neg_criterion, x0=center, **options)

        critval = _eval_absolute_value(
            result_max.x, intercept, linear_terms, square_terms
        )

        if critval > lambda_:
            lambda_ = critval
            argmax = result_max.x
            idx_max = idx

    return lambda_, argmax, idx_max


def improve_poisedness(sample, shape="sphere", maxiter=5):
    """Improve the lambda poisedness of the sample.

    The poisedness of the sample is improved in an incremental manner; replacing
    one point at a time and reducing the upper bound on the absolute value of
    the Lagrange polynomial.

    The implementation is based on algorithm 6.3 in :cite:`Conn2009`,
    Chapter 6, p. 95 ff.

    Args:
        sample (np.ndarry): Array of shape (n_samples, n_params).
        shape (str): Geometric shape of the sample space. One of "sphere", "cube".
            Default is "sphere".
        maxiter (int): Maximum number of replacement iterations. Default is 5.

    Returns:
        tuple:
            - sample_improved (np.ndarray): Sample with improved poisedness.
            - lambdas (list): History of lambdas.

    """
    sample_improved = sample.copy()

    lambdas = []

    for _ in range(maxiter):
        lambda_, argmax, idx_max = get_poisedness_constant(
            sample=sample_improved, shape=shape
        )

        lambdas += [lambda_]
        sample_improved[idx_max] = argmax

    return sample_improved, lambdas


def _lagrange_poly_matrix(sample):
    """Construct matrix of lagrange polynomials.

    See :cite:`Conn2009`, Chapter 4.2, p. 60.

    Args:
        sample (np.ndarry): Array of shape (n_samples, n_params).

    Returns:
        np.ndarray: Matrix of lagrange polynomials of shape
            (n_samples, n_params * (n_params + 1) // 2).

    """
    basis_mat = _scaled_polynomial_features(sample)
    lagrange_mat = basis_mat @ np.linalg.pinv(basis_mat.T @ basis_mat)

    return lagrange_mat


def _scaled_polynomial_features(x):
    """Construct linear terms, interactions, and scaled square terms.

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


def _get_minimize_options(shape, n_params):
    """Get the minimizer options."""
    if shape == "sphere":
        nonlinear_constraint = NonlinearConstraint(lambda x: np.linalg.norm(x), 0, 1)
        options = {"method": "trust-constr", "constraints": [nonlinear_constraint]}

    elif shape == "cube":
        bound_constraints = Bounds(-np.ones(n_params), np.ones(n_params))
        options = {"method": "trust-constr", "bounds": bound_constraints}

    else:
        raise ValueError(
            f"Invalid shape argument: {shape}. Must be one of sphere, cube."
        )

    return options


def _eval_absolute_value(x, intercept, linear_terms, square_terms):
    return np.abs(intercept + linear_terms.T @ x + 0.5 * x.T @ square_terms @ x)


def _eval_neg_absolute_value(x, intercept, linear_terms, square_terms):
    return -_eval_absolute_value(x, intercept, linear_terms, square_terms)
