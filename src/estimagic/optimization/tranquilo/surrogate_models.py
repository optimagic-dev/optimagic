import inspect
import warnings
from functools import partial

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def get_fitter(fitter, fitter_options=None, data_options=None):
    """Get a fit-function with partialled options.

    Args:
        fitter (str or callable): Name of a fit method or a fit method. The first
            argument of any fit method needs to be ``x``, the second ``y``.

        fitter_options (dict): Options for the fit method. The following are supported:
            - l2_penalty (float): Value to be used for the l2 penalty.

        data_options (dict): Options for the creation of the feature matrix. The
            following are supported:
            - fit_intercept (bool): Whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (i.e. data is
            expected to be centered).
            - quadratic_terms (bool): Whether to use quadratic terms as features in the
            regression.
            - interaction_terms (bool): Whether to use interaction terms as features in
            the regression.

    Returns:
        callable: The fit method.

    """
    fitter_options = fitter_options or {}
    data_options = data_options or {}

    built_in_fitters = {"ols": fit_ols, "ridge": fit_ridge}

    if isinstance(fitter, str) and fitter in built_in_fitters:
        _fitter = built_in_fitters[fitter]
        _fitter_name = fitter
    elif callable(fitter):
        _fitter = fitter
        _fitter_name = getattr(fitter, "__name__", "your solver")
    else:
        raise ValueError(
            "Invalid fitter: {fitter}. Must be one of "
            f"{list(built_in_fitters)} or a callable."
        )

    # consolidate fitter options
    default_fitter_options = {"l2_penalty": 0.1}

    all_options = {**default_fitter_options, **fitter_options}

    args = set(inspect.signature(_fitter).parameters)

    if not {"x", "y"}.issubset(args):
        raise ValueError("fit method needs to take 'x','y' as the first two arguments.")

    not_options = {"x", "y"}
    if isinstance(_fitter, partial):
        partialed_in = set(_fitter.args).union(set(_fitter.keywords))
        not_options = not_options | partialed_in

    valid_options = args - not_options

    reduced = {key: val for key, val in all_options.items() if key in valid_options}

    ignored = {
        key: val for key, val in fitter_options.items() if key not in valid_options
    }

    if ignored:
        warnings.warn(
            "The following options were ignored because they are not compatible "
            f"with {_fitter_name}:\n\n {ignored}"
        )

    # consolidate data options
    default_data_options = {
        "fit_intercept": True,
        "quadratic_terms": True,
        "interaction_terms": True,
    }
    all_options = {**default_data_options, **data_options}
    valid_options = set(default_data_options.keys())

    data_options = {
        key: val for key, val in all_options.items() if key in valid_options
    }

    ignored = {key: val for key, val in all_options.items() if key not in valid_options}

    if ignored:
        warnings.warn(
            "The following options were ignored because they are not compatible "
            f"with the data options:\n\n {ignored}"
        )

    out = partial(
        _fitter_template, fitter=_fitter, fitter_options=reduced, **data_options
    )
    return out


def _fitter_template(
    centered_x,
    centered_f,
    fitter,
    fitter_options,
    fit_intercept,
    quadratic_terms,
    interaction_terms,
):
    """Fit a model to the data.

    Args:
        fitter (str or callable): Name of a fit method or a fit method. The first
            argument of any fit method needs to be ``x``, the second ``f``.
        fitter_options (dict): Options for the fit method. The following are supported:
            - l2_penalty (float or np.ndarray): Value to be used for the l2 penalty.
        fit_intercept (bool): Whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations (i.e. data is expected
            to be centered).
        quadratic_terms (bool): Whether to use quadratic terms as features in the
            regression.
        interaction_terms (bool): Whether to use interaction terms as features in the
            regression.

    Returns:
        (dict): Result dictionary containing the following entries:
            - "intercepts" np.ndarray of shape (n_residuals, 1)
            - "linear_terms" np.ndarray of shape (n_residuals, n_params)
            - "square_terms" np.ndarray of shape (n_residuals, n_params, n_params)

    """
    n_params = centered_x.shape[1]
    n_residuals = centered_f.shape[1]

    x = _build_feature_matrix(
        centered_x, fit_intercept, quadratic_terms, interaction_terms
    )

    fitter_options = _update_penalty_args(fitter_options, x, fit_intercept)

    coef = fitter(x, centered_f, **fitter_options)

    if fit_intercept:
        intercepts, linear_terms, square_terms = np.split(
            coef, (1, n_params + 1), axis=1
        )
    else:
        intercepts = np.zeros((n_residuals, 1))
        linear_terms, square_terms = np.split(coef, (n_params,), axis=1)

    # results processing
    out = {"intercepts": intercepts, "linear_terms": linear_terms}

    if interaction_terms:
        triu = _reshape_square_terms_to_triu(
            square_terms, n_params, n_residuals, quadratic_terms
        )
        out["square_terms"] = triu + triu.transpose(0, 2, 1)
    elif quadratic_terms:
        out["square_terms"] = np.concatenate(
            [np.diag(2 * a)[np.newaxis] for a in list(square_terms)], axis=0
        )
    else:
        out["square_terms"] = np.zeros((n_residuals, n_params, n_params))

    return out


def fit_ols(x, y):
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


def fit_ridge(x, y, l2_penalty):
    """Fit a linear model using ridge regression.

    Args:
        x (np.ndarray): Array of shape (n, p) of x-values.
        y (np.ndarray): Array of shape (n, k) of y-values.
        l2_penalty (np.ndarray): Array of shape (p, ) of penalty values.

    Returns:
        coef (np.ndarray): Array of shape (p, k) of coefficients.

    """
    a = x.T @ x
    a = np.fill_diagonal(a, a.diagonal() + l2_penalty)
    b = x.T @ y

    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    coef = coef.T
    return coef


def _build_feature_matrix(x, fit_intercept, quadratic_terms, interaction_terms):
    if interaction_terms:
        poly = PolynomialFeatures(
            degree=2, include_bias=fit_intercept, interaction_only=not quadratic_terms
        )
        features = poly.fit_transform(x)
    else:
        data = (np.ones(len(x)), x) if fit_intercept else (x,)
        data = (*data, x**2) if quadratic_terms else data
        features = np.column_stack(data)
    return features


def _reshape_square_terms_to_triu(square_terms, n_params, n_residuals, quadratic_terms):
    offset = 0 if quadratic_terms else 1
    idx1, idx2 = np.triu_indices(n_params, k=offset)
    triu = np.zeros((n_residuals, n_params, n_params), dtype=np.float64)
    triu[:, idx1, idx2] = square_terms
    return triu


def _update_penalty_args(fitter_options, features, fit_intercept):
    options = fitter_options.copy()
    penalty_args = {"l2_penalty"}
    for arg in penalty_args.intersection(options.keys()):
        penalty = options[arg]
        penalty = penalty * np.ones(features.shape[1])
        if fit_intercept:
            penalty[0] = 0
        options[arg] = penalty
    return options
