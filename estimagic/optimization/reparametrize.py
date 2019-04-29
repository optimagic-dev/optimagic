"""Handle constraints by bounds and reparametrizations."""

import numpy as np
import warnings
from estimagic.optimization.utilities import (
    cov_params_to_matrix,
    number_of_triangular_elements_to_dimension,
)


def reparametrize_to_internal(params, constraints):
    """Convert a params DataFrame to an internal_params DataFrame.

    The internal params df is shorter because it does not contain fixed parameters.
    Moreover, it contains a reparametrized 'value' column that can be used to construct
    a parameter vector that satisfies all constraints.

    Args:
        params (DataFrame): A non-internal parameter DataFrame. See :ref:`params_df`.


    """
    internal = params.copy()
    for constr in constraints:
        params_subset = params.loc[constr['selector']]
        if constr['type'] == 'covariance':
            internal.update(_covariance_to_internal(params_subset))
        elif constr['type'] == 'sum':
            internal.update(_sum_to_internal(params_subset, constr['value']))
        elif constr['type'] == 'probability':
            internal.update(_probability_to_internal(params_subset))
        elif constr['type'] == 'increasing':
            internal.update(_increasing_to_internal(params_subset))
        elif constr['type'] == 'equality':
            internal.update(_equality_to_internal(params_subset))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="indexing past lexsort depth may impact performance.")
        actually_fixed = internal['lower'] == internal['upper']
        internal.loc[actually_fixed, 'fixed'] = True

    internal = internal.query('~fixed')

    return internal


def reparametrize_from_internal(internal_params, constraints, original_params):
    """Convert an internal_params DataFrame to a Series with valid parameters.

    The parameter values are constructed from the 'value' column of internal_params.
    The resulting Series has the same index as the non-internal params DataFrame.

    """
    reindexed = internal_params.reindex(original_params.index)
    params_sr = reindexed.copy()
    fixed_index = params_sr[params_sr.isnull()].index

    # writing the fixed parameters back has to be done before all other constraints
    # are handled!
    params_sr[fixed_index] = original_params.loc[fixed_index, 'value']

    for constr in constraints:
        params_subset = reindexed.loc[constr['selector']]
        if constr['type'] == 'covariance':
            params_sr.update(_covariance_to_internal(params_subset))
        elif constr['type'] == 'sum':
            params_sr.update(_sum_to_internal(params_subset, constr['value']))
        elif constr['type'] == 'probability':
            params_sr.update(_probability_to_internal(params_subset))
        elif constr['type'] == 'increasing':
            params_sr.update(_increasing_to_internal(params_subset))
        elif constr['type'] == 'equality':
            params_sr.update(_equality_to_internal(params_subset))

    return params_sr


def _covariance_to_internal(params_subset):
    """Ensure valid covariance matrices.

    The parameters in params_subset are assumed to be the lower triangular elements of
    a covariance matrix.

    If all off-diagonal elements are fixed to zero, it is only necessary to set the
    lower bounds to 0, unless already stricter. Otherwise, we do a (lower triangular)
    Cholesky reparametrization and restrict diagonal elements to be positive (see:
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.494&rep=rep1&type=pdf)

    Note that the cholesky reparametrization is not compatible with any other
    constraints on the involved parameters. Moreover, it requires the covariance matrix
    described by the start values to be positive definite as opposed to positive
    semi-definite.

    """
    cov = cov_params_to_matrix(params_subset["value"].to_numpy())
    dim = len(cov)
    off_diagonal_zero = bool((cov[np.tril_indices(dim, k=-1)] == 0).all())

    fixed_helper = cov_params_to_matrix(params_subset["fixed"].to_numpy()).astype(bool)
    off_diagonal_fixed = bool(fixed_helper[np.tril_indices(dim, k=-1)].all())
    all_fixed = bool(params_subset["fixed"].all())

    res = params_subset.copy()

    e, v = np.linalg.eigh(cov)
    assert np.all(e > -1e-8), "Invalid covariance matrix."

    if all_fixed:
        pass
    elif off_diagonal_fixed and off_diagonal_zero:
        lower_bound_helper = cov_params_to_matrix(params_subset["lower"])
        diag_lower = np.maximum(np.diagonal(lower_bound_helper), np.zeros(dim))
        lower_bound_helper[np.diag_indices(dim)] = diag_lower
        lower_bounds = lower_bound_helper[np.tril_indices(dim)]

        res["lower"] = lower_bounds

        assert (res["upper"] >= res["lower"]).all(), "Invalid upper bound for variance."
    else:
        chol = np.linalg.cholesky(cov)
        chol_coeffs = chol[np.tril_indices(dim)]
        res["value"] = chol_coeffs

        res["lower"] = -np.inf
        res["upper"] = np.inf
        res["fixed"] = False

        if params_subset["fixed"].any():
            warnings.warn("Covariance parameters are unfixed.", UserWarning)

        for bound in ["lower", "upper"]:
            if np.isfinite(params_subset[bound]).any():
                warnings.warn(
                    "Bounds are ignored for covariance parameters.", UserWarning
                )

    return res


def _covariance_from_internal(params_subset):
    res = params_subset.copy(deep=True)
    dim = number_of_triangular_elements_to_dimension(len(params_subset))
    helper = np.zeros((dim, dim))
    helper[np.tril_indices(dim)] = params_subset["value"].to_numpy()
    cov = helper.dot(helper.T)
    cov_coeffs = cov[np.tril_indices(dim)]
    res["value"] = cov_coeffs
    return res


def _increasing_to_internal(params_subset):
    """Ensure that the parameters in params_subset are increasing."""
    old_vals = params_subset["value"].to_numpy()
    new_vals = old_vals.copy()
    new_vals[1:] -= old_vals[:-1]
    res = params_subset.copy()
    res["value"] = new_vals

    res["fixed"] = False
    res["lower"] = -np.inf
    res["upper"] = np.inf

    if params_subset["fixed"].any():
        warnings.warn("Ordered parameters were unfixed.", UserWarning)

    for bound in ["lower", "upper"]:
        if np.isfinite(params_subset[bound]).any():
            warnings.warn("Bounds are ignored for ordered parameters.", UserWarning)

    return res


def _increasing_from_internal(params_subset):
    res = params_subset.copy()
    res["value"] = params_subset["value"].cumsum()
    return res


def _sum_to_internal(params_subset, value):
    free = params_subset.query("lower == -inf & upper == inf & fixed == False")
    last = params_subset.index[-1]

    assert (
        last in free.index
    ), "The last sum constrained parameter cannot have bounds nor be fixed."

    res = params_subset.copy()
    res.loc[last, "fixed"] = True
    return res


def _sum_from_internal(params_subset, value):
    res = params_subset.copy()
    last = params_subset.index[-1]
    all_others = params_subset.index[:-1]
    res.loc[last, "value"] = value - params_subset.loc[all_others, "value"].sum()
    return res


def _probability_to_internal(params_subset):
    res = params_subset.copy()
    assert (
        params_subset["lower"].isin([-np.inf, 0]).all()
    ), "Lower bound has to be 0 or -inf for probability constrained parameters."

    assert (
        params_subset["upper"].isin([np.inf, 1]).all()
    ), "Upper bound has to be 1 or inf for probability constrained parameters."

    assert not params_subset[
        "fixed"
    ].any(), "Probability constrained parameters cannot be fixed."

    res["lower"] = -np.inf
    res["upper"] = np.inf
    last = params_subset.index[-1]
    res.loc[last, "fixed"] = True
    res['value'] /= res.loc[last, 'value']
    return res


def _probability_from_internal(params_subset):
    last = params_subset.index[-1]
    res = params_subset.copy()
    res.loc[last, 'value'] = 1
    res["value"] /= params_subset["value"].sum()
    return res


def _equality_to_internal(params_subset):
    res = params_subset.copy()
    first = params_subset.index[0]
    all_others = params_subset.index[1:]
    res.loc[all_others, "value"] = res.loc[first, "value"]
    res.loc[first, "fixed"] = params_subset["fixed"].any()
    res.loc[all_others, "fixed"] = True
    res["lower"] = params_subset["lower"].max()
    res["upper"] = params_subset["upper"].min()
    assert (
        res["lower"] <= res["upper"]
    ).all(), "Invalid bounds for equality constrained parameters."
    return res


def _equality_from_internal(params_subset):
    res = params_subset.copy()
    first = params_subset.index[0]
    all_others = params_subset.index[1:]
    res.loc[all_others, "value"] = res.loc[first, "value"]
    return res
