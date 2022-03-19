import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def fit_ols(centered_x, centered_f, *, fit_intercept=True, interaction_terms=True):
    """Fit a quadratic surrogate model with ols.

    Args:
        centered_x (np.ndarray): Array of shape (n_sample, n_params) of x-values,
            rescaled such that the trust region becomes a hypercube from -1 to 1.
        centered_f (np.ndarray): Array of shape (n_sample, n_residuals) with function
            evaluations that have been centered around the function value at the
            trust region center.
        fit_intercept (bool): Whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations (i.e. data is expected
            to be centered). Default is True.
        interaction_terms (bool): Whether to use interaction terms as features in the
            regression. Default is True.

    Returns:
        dict: Dictionary containing the parameters of the residual model with keys
            "linear_terms" and "square_terms". The linear terms are returned as a
            1d array. If interaction_terms is False the squared terms are also returned
            as a 1d array, otherwise as a lower triangular matrix.

    """
    n_params = centered_x.shape[1]
    n_residuals = centered_f.shape[1]

    x = _build_quadratic_feature_matrix(centered_x, fit_intercept, interaction_terms)

    # compute coefficients
    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx)

    coef = xtx_inv @ x.T @ centered_f

    # extract terms
    _, linear_terms, square_terms = np.split(coef, (1, n_params + 1), axis=0)

    out = {"linear_terms": linear_terms}

    if interaction_terms:
        out["square_terms"] = _reshape_square_terms_to_tril(
            square_terms, n_params, n_residuals
        )
    else:
        out["square_terms"] = square_terms

    return out


def _build_quadratic_feature_matrix(x, fit_intercept, interaction_terms):
    if interaction_terms:
        poly = PolynomialFeatures(degree=2, include_bias=fit_intercept)
        xx = poly.fit_transform(x)
    else:
        xx = np.column_stack((x, x**2))
        xx = np.column_stack((np.ones(len(x)), xx)) if fit_intercept else xx
    return xx


def _reshape_square_terms_to_tril(square_terms, n_params, n_residuals):
    n_terms = int(np.sqrt(1 / 4 + 2 * len(square_terms) - 1 / 2))
    idx1, idx2 = np.triu_indices(n_terms)
    triu = np.zeros((n_terms, n_terms, n_residuals), dtype=np.float64)
    triu[idx1, idx2, :] = square_terms
    tril = triu.transpose((1, 0, 2))
    return tril
