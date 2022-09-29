import inspect
import warnings
from functools import partial

import numpy as np
from estimagic.optimization.tranquilo.models import ModelInfo
from estimagic.optimization.tranquilo.models import VectorModel
from numba import njit


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
            - has_intercepts (bool): Whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (i.e. data is
            expected to be centered).
            - has_squares (bool): Whether to use quadratic terms as features in the
            regression.
            - has_interactions (bool): Whether to use interaction terms as features
            in the regression.

    Returns:
        callable: The partialled fit method that only depends on x and y.
    """
    user_options = user_options or {}
    model_info = model_info or ModelInfo()

    built_in_fitters = {"ols": fit_ols, "ridge": fit_ridge, "pounders": fit_pounders}

    if isinstance(fitter, str) and fitter in built_in_fitters:
        _fitter = built_in_fitters[fitter]
        _fitter_name = fitter
    elif callable(fitter):
        _fitter = fitter
        _fitter_name = getattr(fitter, "__name__", "your fitter")
    else:
        raise ValueError(
            f"Invalid fitter: {fitter}. Must be one of {list(built_in_fitters)} or a "
            "callable."
        )

    default_options = {
        "l2_penalty_linear": 0,
        "l2_penalty_square": 0.1,
    }

    all_options = {**default_options, **user_options}

    args = set(inspect.signature(_fitter).parameters)

    if not {"x", "y", "model_info"}.issubset(args):
        raise ValueError(
            "fit method needs to take 'x', 'y' and 'model_info' as the first three "
            "arguments."
        )

    not_options = {"x", "y", "model_info"}
    if isinstance(_fitter, partial):
        partialed_in = set(_fitter.args).union(set(_fitter.keywords))
        not_options = not_options | partialed_in

    valid_options = args - not_options

    reduced = {key: val for key, val in all_options.items() if key in valid_options}

    ignored = {
        key: val for key, val in user_options.items() if key not in valid_options
    }

    if ignored:
        warnings.warn(
            "The following options were ignored because they are not compatible with "
            f"{_fitter_name}:\n\n {ignored}"
        )

    out = partial(
        _fitter_template, fitter=_fitter, model_info=model_info, options=reduced
    )

    return out


def _fitter_template(
    x,
    y,
    fitter,
    model_info,
    options,
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
        options (dict): Options for the fit method.

    Returns:
        VectorModel or ScalarModel: Results container.
    """
    n_params = x.shape[1]
    n_residuals = y.shape[1]

    coef = fitter(x, y, model_info, **options)

    # results processing
    if model_info.has_intercepts:
        intercepts, linear_terms, square_terms = np.split(
            coef, (1, n_params + 1), axis=1
        )
        intercepts = intercepts.flatten()
    else:
        intercepts = None
        linear_terms, square_terms = np.split(coef, (n_params,), axis=1)

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
    cutoffs = (1, n_params + 1) if model_info.has_intercepts else (0, n_params)

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


def fit_pounders(x, y, model_info):
    """Fit a linear model using the pounders fitting method.

    The solution represents the quadratic whose Hessian matrix is of
    minimum Frobenius norm.

    For a mathematical exposition, see :cite:`Wild2008`, p. 3-5.

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
    _is_just_identified = n_samples == (n_params + 1)
    has_intercepts = model_info.has_intercepts
    has_squares = model_info.has_squares

    features = _polynomial_features(x, has_intercepts, has_squares)
    m_mat, m_mat_pad, n_mat = _build_feature_matrices_pounders(
        features, n_params, n_samples, has_intercepts
    )
    z_mat = _calculate_basis_null_space(m_mat_pad, n_samples, n_params)
    n_z_mat = _multiply_feature_matrix_with_basis_null_space(
        n_mat, z_mat, n_samples, n_params, _is_just_identified
    )

    coef = _get_current_fit_pounders(
        y,
        m_mat,
        n_mat,
        z_mat,
        n_z_mat,
        n_params,
        has_intercepts,
        _is_just_identified,
    )

    return coef


def _get_current_fit_pounders(
    y,
    m_mat,
    n_mat,
    z_mat,
    n_z_mat,
    n_params,
    has_intercepts,
    _is_just_identified,
):
    n_residuals = y.shape[1]
    n_poly_features = n_params * (n_params + 1) // 2
    offset = 0 if has_intercepts else 1

    coef = np.empty((n_residuals, has_intercepts + n_params + n_poly_features))

    if _is_just_identified:
        coeffs_first_stage = np.zeros(n_params)
        coeffs_square = np.zeros(n_poly_features)

    for resid in range(n_residuals):
        if not _is_just_identified:
            z_y_vec = z_mat.T @ y[:, resid]
            coeffs_first_stage = np.linalg.solve(
                np.atleast_2d(n_z_mat.T @ n_z_mat),
                np.atleast_1d(z_y_vec),
            )
            coeffs_square = np.atleast_2d(n_z_mat) @ coeffs_first_stage

        rhs = y[:, resid] - n_mat @ coeffs_square
        coeffs_linear = np.linalg.solve(m_mat, rhs[: n_params + 1])

        coef[resid] = np.concatenate((coeffs_linear[offset:], coeffs_square), axis=None)

    return coef


def _build_feature_matrices_pounders(features, n_params, n_samples, has_intercepts):
    m_mat, n_mat = np.split(features, (n_params + has_intercepts,), axis=1)
    m_mat_pad = np.zeros((n_samples, n_samples))
    m_mat_pad[:, : n_params + has_intercepts] = m_mat

    return m_mat[: n_params + 1, : n_params + 1], m_mat_pad, n_mat


def _multiply_feature_matrix_with_basis_null_space(
    n_mat, z_mat, n_samples, n_params, _is_just_identified
):
    n_z_mat = n_mat.T @ z_mat

    if _is_just_identified:
        n_z_mat_pad = np.zeros((n_samples, (n_params * (n_params + 1) // 2)) + 1)
        n_z_mat_pad[:n_params, :n_params] = np.eye(n_params)
        n_z_mat = n_z_mat_pad[:, n_params + 1 : n_samples]

    return n_z_mat


def _calculate_basis_null_space(m_mat_pad, n_samples, n_params):
    q_mat, _ = np.linalg.qr(m_mat_pad)

    return q_mat[:, n_params + 1 : n_samples]


def _build_feature_matrix(x, model_info):
    if model_info.has_interactions:
        features = _polynomial_features(
            x, model_info.has_intercepts, model_info.has_squares
        )
    else:
        data = (np.ones(len(x)), x) if model_info.has_intercepts else (x,)
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
