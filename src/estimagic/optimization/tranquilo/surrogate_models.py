import inspect
import warnings
from functools import partial

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def get_fitter(fitter, user_options=None):
    """Get a fit-function with partialled options.

    Args:
        fitter (str or callable): Name of a fit method or a fit method. The first
            argument of any fit method needs to be ``x``, the second ``y``.

        user_options (dict): Options for the fit method. The following are supported:
            - l2_penalty_linear (float): Penalty that is applied to all linear terms.
            - l2_penalty_square (float): Penalty that is applied to all square terms,
            that is the quadratic and interaction terms.
            - fit_intercept (bool): Whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (i.e. data is
            expected to be centered).
            - include_squares (bool): Whether to use quadratic terms as features in the
            regression.
            - include_interaction (bool): Whether to use interaction terms as features
            in the regression.

    Returns:
        callable: The fit method.

    """
    user_options = user_options or {}

    built_in_fitters = {"ols": fit_ols, "ridge": fit_ridge}

    if isinstance(fitter, str) and fitter in built_in_fitters:
        _fitter = built_in_fitters[fitter]
        _fitter_name = fitter
    elif callable(fitter):
        _fitter = fitter
        _fitter_name = getattr(fitter, "__name__", "your solver")
    else:
        raise ValueError(
            f"Invalid fitter: {fitter}. Must be one of {list(built_in_fitters)} or a "
            "callable."
        )

    default_options = {
        "fit_intercept": True,
        "include_squares": True,
        "include_interaction": True,
        "l2_penalty_linear": 0,
        "l2_penalty_square": 0.1,
    }

    all_options = {**default_options, **user_options}

    args = set(inspect.signature(_fitter).parameters)

    if not {"x", "y"}.issubset(args):
        raise ValueError(
            "fit method needs to take 'x' and 'y' as the first two arguments."
        )

    not_options = {"x", "y"}
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
            "The following options were ignored because they are not compatible "
            f"with {_fitter_name}:\n\n {ignored}"
        )

    out = partial(_fitter_template, fitter=_fitter, options=reduced)
    return out


def _fitter_template(
    x,
    y,
    fitter,
    options,
):
    """Fit a model to data.

    Args:
        fitter (str or callable): Name of a fit method or a fit method. The first
            argument of any fit method needs to be ``x``, the second ``f``.
        options (dict): Options for the fit method. The following are supported:
            - l2_penalty (float): Value to be used for the l2 penalty.
            - fit_intercept (bool): Whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (i.e. data is
            expected to be centered).
            - include_squares (bool): Whether to use quadratic terms as features in the
            regression.
            - include_interaction (bool): Whether to use interaction terms as features
            in the regression.

    Returns:
        (dict): Result dictionary containing the following entries:
            - "intercepts" np.ndarray of shape (n_residuals, 1)
            - "linear_terms" np.ndarray of shape (n_residuals, n_params)
            - "square_terms" np.ndarray of shape (n_residuals, n_params, n_params)

    """
    coef = fitter(x, y, **options)

    n_params = x.shape[1]
    n_residuals = y.shape[1]

    fit_intercept, include_squares, include_interaction = _get_data_options(options)

    if fit_intercept:
        intercepts, linear_terms, square_terms = np.split(
            coef, (1, n_params + 1), axis=1
        )
    else:
        intercepts = np.zeros((n_residuals, 1))
        linear_terms, square_terms = np.split(coef, (n_params,), axis=1)

    # results processing
    out = {"intercepts": intercepts, "linear_terms": linear_terms}

    if include_interaction:
        triu = _reshape_square_terms_to_triu(
            square_terms, n_params, n_residuals, include_squares
        )
        out["square_terms"] = triu + triu.transpose(0, 2, 1)
    elif include_squares:
        out["square_terms"] = np.concatenate(
            [np.diag(2 * a)[np.newaxis] for a in list(square_terms)], axis=0
        )
    else:
        out["square_terms"] = np.zeros((n_residuals, n_params, n_params))

    return out


def fit_ols(x, y, fit_intercept, include_squares, include_interaction):
    """Fit a linear model using ordinary least squares.

    Args:
        x (np.ndarray): Array of shape (n_samples, n_params) of x-values,
            rescaled such that the trust region becomes a hypercube from -1 to 1.
        y (np.ndarray): Array of shape (n_samples, n_residuals) with function
            evaluations that have been centered around the function value at the
            trust region center.
        fit_intercept (bool): Whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations (i.e. data is expected
            to be centered).
        include_squares (bool): Whether to use squared terms as features in the
            regression.
        include_interaction (bool): Whether to use interaction terms as features in the
            regression.

    Returns:
        np.ndarray: The model coefficients.

    """
    features = _build_feature_matrix(
        x, fit_intercept, include_squares, include_interaction
    )
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
    fit_intercept,
    include_squares,
    include_interaction,
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
        fit_intercept (bool): Whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations (i.e. data is expected
            to be centered).
        include_squares (bool): Whether to use squared terms as features in the
            regression.
        include_interaction (bool): Whether to use interaction terms as features in the
            regression.
        l2_penalty_linear (float): Penalty that is applied to all linear terms.
        l2_penalty_square (float): Penalty that is applied to all square terms, that is
            the quadratic and interaction terms.

    Returns:
        np.ndarray: The model coefficients.

    """
    features = _build_feature_matrix(
        x, fit_intercept, include_squares, include_interaction
    )

    # create penalty array
    n_params = x.shape[1]
    cutoffs = (1, n_params + 1) if fit_intercept else (0, n_params)

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
        coef (np.ndarray): Array of shape (p, k) of coefficients.

    """
    a = x.T @ x
    b = x.T @ y

    coef, *_ = np.linalg.lstsq(a + np.diag(penalty), b, rcond=None)
    coef = coef.T
    return coef


def _build_feature_matrix(x, fit_intercept, include_squares, include_interaction):
    if include_interaction:
        poly = PolynomialFeatures(
            degree=2, include_bias=fit_intercept, interaction_only=not include_squares
        )
        features = poly.fit_transform(x)
    else:
        data = (np.ones(len(x)), x) if fit_intercept else (x,)
        data = (*data, x**2) if include_squares else data
        features = np.column_stack(data)
    return features


def _reshape_square_terms_to_triu(square_terms, n_params, n_residuals, include_squares):
    offset = 0 if include_squares else 1
    idx1, idx2 = np.triu_indices(n_params, k=offset)
    triu = np.zeros((n_residuals, n_params, n_params), dtype=np.float64)
    triu[:, idx1, idx2] = square_terms
    return triu


def _get_data_options(options):
    data_options = ["fit_intercept", "include_squares", "include_interaction"]
    out = tuple(options[key] for key in data_options)
    return out
