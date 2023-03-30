from functools import partial

import numpy as np
from numba import njit
from scipy.linalg import qr_multiply

from estimagic.optimization.tranquilo.get_component import get_component
from estimagic.optimization.tranquilo.handle_infinity import get_infinity_handler
from estimagic.optimization.tranquilo.options import FitterOptions
from estimagic.optimization.tranquilo.models import (
    VectorModel,
    add_models,
    move_model,
    n_second_order_terms,
)


def get_fitter(
    fitter,
    fitter_options=None,
    model_type=None,
    residualize=None,
    infinity_handling=None,
):
    """Get a fit-function with partialled options.

    Args:
        fitter (str or callable): Name of a fit method or a fit method. Arguments need
            to be, in order,
            - x (np.ndarray): Data points.
            - y (np.ndarray): Corresponding function evaluations at data points.
            - weighs (np.ndarray): Weights for the data points.
            - model_type (str): Type of model to be fitted.

        fitter_options (dict): Options for the fit method. The following are supported:
            - l2_penalty_linear (float): Penalty that is applied to all linear terms.
            - l2_penalty_square (float): Penalty that is applied to all square terms,
            that is the quadratic and interaction terms.

        model_type (str): Type of the model that is fitted. The following are supported:
            - "linear": Only linear effects and intercept.
            - "quadratic": Fully quadratic model.

        residualize (bool): If True, the model is fitted to the residuals of the old
            model. This introduces momentum when the coefficients are penalized.

        infinity_handling (str): How to handle infinite values in the data. Currently
            supported: {"relative"}. See `handle_infinty.py`.

    Returns:
        callable: The partialled fit method that only depends on x and y.

    """
    built_in_fitters = {
        "ols": fit_ols,
        "ridge": fit_ridge,
        "powell": fit_powell,
        "tranquilo": fit_tranquilo,
    }

    mandatory_arguments = ["x", "y", "model_type"]

    _raw_fitter = get_component(
        name_or_func=fitter,
        component_name="fitter",
        func_dict=built_in_fitters,
        default_options=FitterOptions(),
        user_options=fitter_options,
        mandatory_signature=mandatory_arguments,
    )

    clip_infinite_values = get_infinity_handler(infinity_handling)

    fitter = partial(
        _fitter_template,
        fitter=_raw_fitter,
        model_type=model_type,
        clip_infinite_values=clip_infinite_values,
        residualize=residualize,
    )

    return fitter


def _fitter_template(
    x,
    y,
    region,
    old_model,
    weights=None,
    fitter=None,
    model_type=None,
    clip_infinite_values=None,
    residualize=False,
):
    """Fit a model to data.

    Args:
        x (np.ndarray): Array of shape (n_samples, n_params) of x-values,
            rescaled such that the trust region becomes a hypercube from -1 to 1.
        y (np.ndarray): Array of shape (n_samples, n_residuals) with function
            evaluations that have been centered around the function value at the
            trust region center.
        fitter (callable): Fit method. The first argument of any fit method needs to be
            ``x``, second ``y`` and third ``model_type``.
        model_type (str): Type of the model that is fitted. The following are supported:
            - "linear": Only linear effects and intercept.
            - "quadratic": Fully quadratic model.

    Returns:
        VectorModel or ScalarModel: Results container.

    """
    _, n_params = x.shape
    n_residuals = y.shape[1]

    y_clipped = clip_infinite_values(y)
    x_unit = region.map_to_unit(x)

    if residualize:
        old_model_moved = move_model(old_model, region)
        y_clipped = y_clipped - old_model_moved.predict(x_unit).reshape(y_clipped.shape)

    coef = fitter(x=x_unit, y=y_clipped, weights=weights, model_type=model_type)

    # results processing
    intercepts, linear_terms, square_terms = np.split(coef, (1, n_params + 1), axis=1)
    intercepts = intercepts.flatten()

    # construct final square terms
    if model_type == "quadratic":
        square_terms = _reshape_square_terms_to_hess(
            square_terms, n_params, n_residuals
        )
    else:
        square_terms = None

    results = VectorModel(
        intercepts,
        linear_terms,
        square_terms,
        shift=region.effective_center,
        scale=region.effective_radius,
    )

    if residualize:
        results = add_models(results, old_model_moved)

    return results


def fit_ols(x, y, weights, model_type):
    """Fit a linear model using ordinary least squares.

    Args:
        x (np.ndarray): Array of shape (n_samples, n_params) of x-values,
            rescaled such that the trust region becomes a hypercube from -1 to 1.
        y (np.ndarray): Array of shape (n_samples, n_residuals) with function
            evaluations that have been centered around the function value at the
            trust region center.
        model_type (str): Type of the model that is fitted. The following are supported:
            - "linear": Only linear effects and intercept.
            - "quadratic": Fully quadratic model.

    Returns:
        np.ndarray: The model coefficients.

    """
    features = _build_feature_matrix(x, model_type)
    features_w, y_w = _add_weighting(features, y, weights)
    coef = _fit_ols(features_w, y_w)

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


def fit_tranquilo(x, y, weights, model_type, p_intercept, p_linear, p_square):
    """Fit a linear model using ordinary least squares.

    The difference to fit_ols is that the linear terms are penalized less strongly
    when the system is underdetermined.

    Args:
        x (np.ndarray): Array of shape (n_samples, n_params) of x-values,
            rescaled such that the trust region becomes a hypercube from -1 to 1.
        y (np.ndarray): Array of shape (n_samples, n_residuals) with function
            evaluations that have been centered around the function value at the
            trust region center.
        model_type (str): Type of the model that is fitted. The following are supported:
            - "linear": Only linear effects and intercept.
            - "quadratic": Fully quadratic model.

    Returns:
        np.ndarray: The model coefficients.

    """
    features = _build_feature_matrix(x, model_type)
    features_w, y_w = _add_weighting(features, y, weights)

    n_params = x.shape[1]
    n_features = features.shape[1]

    factor = np.array(
        [1 / p_intercept]
        + [1 / p_linear] * n_params
        + [1 / p_square] * (n_features - 1 - n_params)
    )

    coef_raw = _fit_ols(features_w * factor, y_w)
    coef = coef_raw * factor

    return coef


def fit_ridge(
    x,
    y,
    weights,
    model_type,
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
        model_type (str): Type of the model that is fitted. The following are supported:
            - "linear": Only linear effects and intercept.
            - "quadratic": Fully quadratic model.
        l2_penalty_linear (float): Penalty that is applied to all linear terms.
        l2_penalty_square (float): Penalty that is applied to all square terms, that is
            the quadratic and interaction terms.

    Returns:
        np.ndarray: The model coefficients.

    """
    features = _build_feature_matrix(x, model_type)

    features_w, y_w = _add_weighting(features, y, weights)

    # create penalty array
    n_params = x.shape[1]
    cutoffs = (1, n_params + 1)

    penalty = np.zeros(features.shape[1])
    penalty[: cutoffs[0]] = 0
    penalty[cutoffs[0] : cutoffs[1]] = l2_penalty_linear
    penalty[cutoffs[1] :] = l2_penalty_square

    coef = _fit_ridge(features_w, y_w, penalty)

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


def fit_powell(x, y, model_type):
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
        model_type (str): Type of the model that is fitted. The following are supported:
            - "linear": Only linear effects and intercept.
            - "quadratic": Fully quadratic model.

    Returns:
        np.ndarray: The model coefficients.

    """
    n_samples, n_params = x.shape

    _switch_to_linear = n_samples <= n_params + 1

    _n_just_identified = n_params + 1
    if model_type == "quadratic":
        _n_just_identified += n_second_order_terms(n_params)

    if _switch_to_linear:
        coef = fit_ols(x, y, weights=None, model_type="linear")
        n_resid, n_present = coef.shape
        padding = np.zeros((n_resid, _n_just_identified - n_present))
        coef = np.hstack([coef, padding])
    elif n_samples >= _n_just_identified:
        coef = fit_ols(x, y, weights=None, model_type=model_type)
    else:
        coef = _fit_minimal_frobenius_norm_of_hessian(x, y)

    return coef


def _fit_minimal_frobenius_norm_of_hessian(x, y):
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

    n_poly_features = n_second_order_terms(n_params)

    (
        m_mat,
        n_mat,
        z_mat,
        n_z_mat,
    ) = _get_feature_matrices_minimal_frobenius_norm_of_hessian(x)

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


def _get_feature_matrices_minimal_frobenius_norm_of_hessian(x):
    n_samples, n_params = x.shape

    intercept = np.ones((n_samples, 1))
    features = np.concatenate((intercept, _quadratic_features(x)), axis=1)
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


def _build_feature_matrix(x, model_type):
    raw = x if model_type == "linear" else _quadratic_features(x)
    intercept = np.ones((len(x), 1))
    features = np.concatenate((intercept, raw), axis=1)
    return features


def _reshape_square_terms_to_hess(square_terms, n_params, n_residuals):
    idx1, idx2 = np.triu_indices(n_params)
    hess = np.zeros((n_residuals, n_params, n_params), dtype=np.float64)
    hess[:, idx1, idx2] = square_terms
    hess = hess + np.triu(hess).transpose(0, 2, 1)

    return hess


@njit
def _quadratic_features(x):
    # Create fully quadratic features without intercept
    n_samples, n_params = x.shape
    n_poly_terms = n_second_order_terms(n_params)

    poly_terms = np.empty((n_poly_terms, n_samples), np.float64)
    xt = x.T

    idx = 0
    for i in range(n_params):
        j_start = i
        for j in range(j_start, n_params):
            poly_terms[idx] = xt[i] * xt[j]
            idx += 1
    out = np.concatenate((xt, poly_terms), axis=0)
    return out.T


def _add_weighting(x, y, weights=None):
    # weight the data in order to get weighted fitting from fitters that do not support
    # weights. Inspired by: https://stackoverflow.com/a/52452833
    n_samples = len(x)
    if weights is not None:
        _root_weights = np.sqrt(weights).reshape(n_samples, 1)
        y = y * _root_weights
        x = x * _root_weights
    return x, y
