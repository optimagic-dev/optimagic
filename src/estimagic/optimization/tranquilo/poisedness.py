import numpy as np
from estimagic.optimization.subsolvers._trsbox_quadratic import minimize_trust_trsbox


def get_lambda_poisedness_constant(sample, lower_bounds, upper_bounds):
    """Calculate the lambda poisedness constant of a sample.

    Lambda-poisedness is a concept to measure how well a set of points is dispersed
    through a region of interest; here the trust-region. The poisedness of a sample
    reflects how well the sample “spans” the region of interest.

    In general, if the sample is lambda-poised with a small lambda, the sample is
    said to have "good" geometry. As lambda grows, the system represented by these
    "points", i.e. vectors, becomes increasingly linearly dependent. The smallest
    possible lambda a sample can achieve is 1: lambda >= 1.

    Formal definition:
    A sample Y is said to be lambda-poised on a region of interest if it is linearly
    independent and the Lagrange polynomials L(i) of points i through N in Y satisfy:

        lambda >= max_i max_x | L(i) |      (1)

    i.e. for each point i in the sample, we maximize the absolute criterion value
    of its lagrange polynomial L(i); we then take the maximum over all these
    criterion values as the lambda constant for this particular sample.

    When we compare different samples on the same trust-region, we are usually
    interested in keeping the sample with the least lambda, so that (1) holds.

    For more details, see Conn et al. (:cite:`Conn2010`).

    Note that the trust-region is centered around the origin, and its radius is
    normalized to 1.

    Args:
        sample (np.ndarray): The data sample of shape (p, n), where p is the number
            of points in the sample and n the number of parameters.
        lower_bounds (np.ndarray): 1d array of shape (n,) with lower bounds for the
            parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n,) with upper bounds for the
            parameter vector x.

    Returns:
        float: The lambda poisedness constant, which is >=1. A small lambda means
            the points in the sample have "good" geometry.

    """
    n_samples, n_params = sample.shape

    interpolation_mat = build_interpolation_matrix(sample.T, n_params, n_samples)
    lambda_ = -999

    for index in range(n_samples):
        linear_terms, square_terms = get_lagrange_polynomial(
            sample=sample,
            interpolation_mat=interpolation_mat,
            index=index,
        )

        x_argmax = minimize_trust_trsbox(
            linear_terms,
            square_terms,
            trustregion_radius=1,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        critval_min = abs(_evaluate_scalar_model(x_argmax, linear_terms, square_terms))

        x_argmin = minimize_trust_trsbox(
            -linear_terms,
            -square_terms,
            trustregion_radius=1,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        critval_max = abs(_evaluate_scalar_model(x_argmin, linear_terms, square_terms))

        if critval_min >= critval_max:
            current_critval = critval_min
        else:
            current_critval = critval_max

        if current_critval > lambda_:
            lambda_ = current_critval

    return lambda_


def get_lagrange_polynomial(
    sample,
    interpolation_mat,
    index,
):
    n_samples, n_params = sample.shape

    if n_samples == n_params:
        rhs = np.zeros(n_params)
        rhs[index] = 1.0
    elif n_samples == (n_params + 1) * (n_params + 2) // 2 - 1:
        rhs = np.zeros(n_samples)
        rhs[index] = 1.0
    else:
        rhs = np.zeros(n_samples + n_params)
        rhs[index] = 1.0

    coef, *_ = np.linalg.lstsq(interpolation_mat, rhs, rcond=None)

    if n_samples == n_params:
        linear_terms = coef
        square_terms = np.zeros((n_params, n_params))
    elif n_samples == (n_params + 1) * (n_params + 2) // 2 - 1:
        linear_terms = coef[:n_params]
        square_terms = _reshape_coef_to_square_terms(coef[n_params:], n_params)
    else:
        linear_terms = coef[n_samples:]
        square_terms = np.zeros((n_params, n_params))
        for i in range(n_samples):
            x = sample[i]
            square_terms += coef[i] * np.outer(x, x)

    return linear_terms, square_terms


def build_interpolation_matrix(sample, n_params, n_samples):
    if n_samples == n_params:
        out = sample.T
    elif n_samples + 1 == (n_params + 1) * (n_params + 2) // 2:
        out = np.empty((n_samples, n_samples))
        out[:, :n_params] = sample.T

        for i in range(n_samples):
            _mat = np.outer(sample[:, i], sample[:, i]) - 0.5 * np.diag(
                sample[:, i] * sample[:, i]
            )
            out[i, n_params:] = _reshape_mat_to_upper_triangular(
                _mat,
                n_params,
            )
    else:
        out = np.zeros((n_samples + n_params, n_samples + n_params))
        for i in range(n_samples):
            for j in range(n_samples):
                out[i, j] = 0.5 * (sample[:, i] @ sample[:, j]) ** 2
        out[:n_samples, n_samples:] = sample.T
        out[n_samples:, :n_samples] = sample

    return out


def _reshape_coef_to_square_terms(vec, n_params):
    mat = np.empty((n_params, n_params))
    idx = -1

    for j in range(n_params):
        for i in range(j + 1):
            idx += 1
            mat[i, j] = vec[idx]
            mat[j, i] = vec[idx]

    return mat


def _reshape_mat_to_upper_triangular(mat, n_params):
    triu = np.empty(n_params * (n_params + 1) // 2)
    idx = -1

    for j in range(n_params):
        for i in range(j + 1):
            idx += 1
            triu[idx] = mat[i, j]

    return triu


def _evaluate_scalar_model(x, linear_terms, square_terms):
    return linear_terms.T @ x + 0.5 * x.T @ square_terms @ x
