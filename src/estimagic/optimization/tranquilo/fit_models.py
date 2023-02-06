from functools import partial

import numpy as np
from numba import njit
from scipy.linalg import qr_multiply

from estimagic.optimization.tranquilo.get_component import get_component
from estimagic.optimization.tranquilo.models import (
    ModelInfo,
    VectorModel,
    n_interactions,
    n_second_order_terms,
)


def get_fitter(fitter, user_options=None, model_info=None):
    """Get a fit-function with partialled options.

    Args:
        fitter (str or callable): Name of a fit method or a fit method. The first
            argument of any fit method needs to be ``x``, second ``y`` and third
            ``model_info``.

        user_options (dict): Options for the fit method. The following are supported:
            - l2_penalty_linear (float): Penalty that is applied to all linear terms.
            - l2_penalty_square (float): Penalty that is applied to all square terms,
            that is the quadratic and interaction terms.

        model_info (ModelInfo): Information that describes the functional form of
            the model. Has entries:
            - has_squares (bool): Whether to use quadratic terms as features in the
            regression.
            - has_interactions (bool): Whether to use interaction terms as features
            in the regression.

    Returns:
        callable: The partialled fit method that only depends on x and y.

    """
    if model_info is None:
        model_info = ModelInfo()

    built_in_fitters = {
        "ols": fit_ols,
        "ridge": fit_ridge,
        "powell": fit_powell,
    }

    default_options = {
        "l2_penalty_linear": 0,
        "l2_penalty_square": 0.1,
        "model_info": model_info,
    }

    mandatory_arguments = ["x", "y", "model_info"]

    _raw_fitter = get_component(
        name_or_func=fitter,
        component_name="fitter",
        func_dict=built_in_fitters,
        default_options=default_options,
        user_options=user_options,
        mandatory_signature=mandatory_arguments,
    )

    fitter = partial(
        _fitter_template,
        fitter=_raw_fitter,
        model_info=model_info,
    )

    return fitter


def _fitter_template(
    x,
    y,
    weights=None,
    fitter=None,
    model_info=None,
):
    """Fit a model to data.

    Args:
        x (np.ndarray): Array of shape (n_samples, n_params) of x-values,
            rescaled such that the trust region becomes a hypercube from -1 to 1.
        y (np.ndarray): Array of shape (n_samples, n_residuals) with function
            evaluations that have been centered around the function value at the
            trust region center.
        fitter (callable): Fit method. The first argument of any fit method needs to be
            ``x``, second ``y`` and third ``model_info``.
        model_info (ModelInfo): Information that describes the functional form of
            the model.

    Returns:
        VectorModel or ScalarModel: Results container.

    """
    n_samples, n_params = x.shape
    n_residuals = y.shape[1]

    # weight the data in order to get weighted fitting from fitters that do not support
    # weights. Inspired by: https://stackoverflow.com/a/52452833
    if weights is not None:
        _root_weights = np.sqrt(weights).reshape(n_samples, 1)
        y = y * _root_weights
        x = x * _root_weights

    coef = fitter(x, y)

    # results processing
    intercepts, linear_terms, square_terms = np.split(coef, (1, n_params + 1), axis=1)
    intercepts = intercepts.flatten()

    # construct final square terms
    if model_info.has_interactions:
        square_terms = _reshape_square_terms_to_hess(
            square_terms, n_params, n_residuals, model_info.has_squares
        )
    elif model_info.has_squares:
        square_terms = 2 * np.stack([np.diag(a) for a in square_terms])
    else:
        square_terms = None

    results = VectorModel(intercepts, linear_terms, square_terms)

    return results


def fit_ols(x, y, model_info):
    """Fit a linear model using ordinary least squares.

    Args:
        x (np.ndarray): Array of shape (n_samples, n_params) of x-values,
            rescaled such that the trust region becomes a hypercube from -1 to 1.
        y (np.ndarray): Array of shape (n_samples, n_residuals) with function
            evaluations that have been centered around the function value at the
            trust region center.
        model_info (ModelInfo): Information that describes the functional form of the
            model.

    Returns:
        np.ndarray: The model coefficients.

    """
    features = _build_feature_matrix(x, model_info)
    coef = _fit_ols(features, y)

    return coef


def _fit_ols(x, y):
    """Fit a linear model using least-squares.

    Args:
        x (np.ndarray): Array of shape (n, p) of x-values.
        y (np.ndarray): Array of shape (n, k) of y-values.

    Returns:
        coef (np.ndarray): Array of shape (p, k) of coefficients.

    """
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    coef = coef.T

    return coef


def fit_ridge(
    x,
    y,
    model_info,
    l2_penalty_linear,
    l2_penalty_square,
):
    """Fit a linear model using Ridge regression.

    Args:
        x (np.ndarray): Array of shape (n_samples, n_params) of x-values, rescaled such
            that the trust region becomes a hypercube from -1 to 1.
        y (np.ndarray): Array of shape (n_samples, n_residuals) with function
            evaluations that have been centered around the function value at the trust
            region center.
        model_info (ModelInfo): Information that describes the functional form of the
            model.
        l2_penalty_linear (float): Penalty that is applied to all linear terms.
        l2_penalty_square (float): Penalty that is applied to all square terms, that is
            the quadratic and interaction terms.

    Returns:
        np.ndarray: The model coefficients.

    """
    features = _build_feature_matrix(x, model_info)

    # create penalty array
    n_params = x.shape[1]
    cutoffs = (1, n_params + 1)

    penalty = np.zeros(features.shape[1])
    penalty[: cutoffs[0]] = 0
    penalty[cutoffs[0] : cutoffs[1]] = l2_penalty_linear
    penalty[cutoffs[1] :] = l2_penalty_square

    coef = _fit_ridge(features, y, penalty)

    return coef


def _fit_ridge(x, y, penalty):
    """Fit a linear model using ridge regression.

    Args:
        x (np.ndarray): Array of shape (n, p) of x-values.
        y (np.ndarray): Array of shape (n, k) of y-values.
        penalty (np.ndarray): Array of shape (p, ) of penalty values.

    Returns:
        np.ndarray: Array of shape (p, k) of coefficients.

    """
    a = x.T @ x
    b = x.T @ y

    coef, *_ = np.linalg.lstsq(a + np.diag(penalty), b, rcond=None)
    coef = coef.T

    return coef


def fit_powell(x, y, model_info):
    """Fit a model, switching between penalized and unpenalized fitting.

    For:
    - n + 1 points: Fit ols with linear feature matrix.
    - n + 2 <= n + 0.5 * n * (n + 1) points, i.e. until one less than a
        just identified quadratic model: Fit pounders.
    - else: Fit ols with quadratic feature matrix.


    Args:
        x (np.ndarray): Array of shape (n_samples, n_params) of x-values,
            rescaled such that the trust region becomes a hypercube from -1 to 1.
        y (np.ndarray): Array of shape (n_samples, n_residuals) with function
            evaluations that have been centered around the function value at the
            trust region center.
        model_info (ModelInfo): Information that describes the functional form of the
            model.

    Returns:
        np.ndarray: The model coefficients.

    """
    n_samples, n_params = x.shape

    _switch_to_linear = n_samples <= n_params + 1

    _n_just_identified = n_params + 1
    if model_info.has_squares:
        _n_just_identified += n_params
    if model_info.has_interactions:
        _n_just_identified += int(0.5 * n_params * (n_params - 1))

    if _switch_to_linear:
        model_info = model_info._replace(has_squares=False, has_interactions=False)
        coef = fit_ols(x, y, model_info)
        n_resid, n_present = coef.shape
        padding = np.zeros((n_resid, _n_just_identified - n_present))
        coef = np.hstack([coef, padding])
    elif n_samples >= _n_just_identified:
        coef = fit_ols(x, y, model_info)
    else:
        coef = _fit_minimal_frobenius_norm_of_hessian(x, y, model_info)

    return coef


def _fit_minimal_frobenius_norm_of_hessian(x, y, model_info):
    """Fit a quadraitc model using the powell fitting method.

    The solution represents the quadratic whose Hessian matrix is of
    minimum Frobenius norm. This has been popularized by Powell and is used in
    many optimizers, e.g. bobyqa and pounders.

    For a mathematical exposition, see :cite:`Wild2008`, p. 3-5.

    This method should only be called if the number of samples is larger than what
    is needed to identify the parameters of a linear model but smaller than what
    is needed to identify the parameters of a quadratic model. Most of the time,
    the sample size is 2n + 1.

    Args:
        x (np.ndarray): Array of shape (n_samples, n_params) of x-values,
            rescaled such that the trust region becomes a hypercube from -1 to 1.
        y (np.ndarray): Array of shape (n_samples, n_residuals) with function
            evaluations that have been centered around the function value at the
            trust region center.
        model_info (ModelInfo): Information that describes the functional form of the
            model.

    Returns:
        np.ndarray: The model coefficients.

    """
    n_samples, n_params = x.shape

    _n_too_few = n_params + 1
    _n_too_many = n_params + n_params * (n_params + 1) // 2 + 1

    if n_samples <= _n_too_few:
        raise ValueError("Too few points for minimum frobenius fitting.")
    if n_samples >= _n_too_many:
        raise ValueError("Too may points for minimum frobenius fitting")

    has_squares = model_info.has_squares

    if has_squares:
        n_poly_features = n_params * (n_params + 1) // 2
    else:
        n_poly_features = n_params * (n_params - 1) // 2

    (
        m_mat,
        n_mat,
        z_mat,
        n_z_mat,
    ) = _get_feature_matrices_minimal_frobenius_norm_of_hessian(x, model_info)

    coef = _get_current_fit_minimal_frobenius_norm_of_hessian(
        y=y,
        m_mat=m_mat,
        n_mat=n_mat,
        z_mat=z_mat,
        n_z_mat=n_z_mat,
        n_params=n_params,
        n_poly_features=n_poly_features,
    )

    return coef


def _get_current_fit_minimal_frobenius_norm_of_hessian(
    y,
    m_mat,
    n_mat,
    z_mat,
    n_z_mat,
    n_params,
    n_poly_features,
):
    n_residuals = y.shape[1]
    offset = 0

    coeffs_linear = np.empty((n_residuals, 1 + n_params))
    coeffs_square = np.empty((n_residuals, n_poly_features))

    n_z_mat_square = n_z_mat.T @ n_z_mat

    for k in range(n_residuals):
        z_y_vec = np.dot(z_mat.T, y[:, k])
        coeffs_first_stage, *_ = np.linalg.lstsq(
            np.atleast_2d(n_z_mat_square), np.atleast_1d(z_y_vec), rcond=None
        )

        coeffs_second_stage = np.atleast_2d(n_z_mat) @ coeffs_first_stage

        rhs = y[:, k] - n_mat @ coeffs_second_stage

        alpha, *_ = np.linalg.lstsq(m_mat, rhs[: n_params + 1], rcond=None)
        coeffs_linear[k, :] = alpha[offset : (n_params + 1)]

        coeffs_square[k] = coeffs_second_stage

    coef = np.concatenate((coeffs_linear, coeffs_square), axis=1)

    return np.atleast_2d(coef)


def _get_feature_matrices_minimal_frobenius_norm_of_hessian(x, model_info):
    n_samples, n_params = x.shape
    has_squares = model_info.has_squares

    features = _polynomial_features(x, has_squares)
    m_mat, n_mat = np.split(features, (n_params + 1,), axis=1)

    m_mat_pad = np.zeros((n_samples, n_samples))
    m_mat_pad[:, : n_params + 1] = m_mat

    n_z_mat, _ = qr_multiply(
        m_mat_pad,
        n_mat.T,
    )

    z_mat, _ = qr_multiply(
        m_mat_pad,
        np.eye(n_samples),
    )

    return (
        m_mat[: n_params + 1, : n_params + 1],
        n_mat,
        z_mat[:, n_params + 1 : n_samples],
        n_z_mat[:, n_params + 1 : n_samples],
    )


def _build_feature_matrix(x, model_info):
    if model_info.has_interactions:
        features = _polynomial_features(x, model_info.has_squares)
    else:
        data = (np.ones(len(x)), x)
        data = (*data, x**2) if model_info.has_squares else data
        features = np.column_stack(data)

    return features


def _reshape_square_terms_to_hess(square_terms, n_params, n_residuals, has_squares):
    offset = 0 if has_squares else 1
    idx1, idx2 = np.triu_indices(n_params, k=offset)
    hess = np.zeros((n_residuals, n_params, n_params), dtype=np.float64)
    hess[:, idx1, idx2] = square_terms
    hess = hess + np.triu(hess).transpose(0, 2, 1)

    return hess


@njit
def _polynomial_features(x, has_squares):
    n_samples, n_params = x.shape

    if has_squares:
        n_poly_terms = n_second_order_terms(n_params)
    else:
        n_poly_terms = n_interactions(n_params)

    poly_terms = np.empty((n_poly_terms, n_samples), np.float64)
    xt = x.T

    idx = 0
    for i in range(n_params):
        j_start = i if has_squares else i + 1
        for j in range(j_start, n_params):
            poly_terms[idx] = xt[i] * xt[j]
            idx += 1

    intercept = np.ones((1, n_samples), x.dtype)
    out = np.concatenate((intercept, xt, poly_terms), axis=0)

    return out.T
