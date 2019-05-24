"""Handle constraints by bounds and reparametrizations."""
import warnings

import numpy as np

from estimagic.optimization.utilities import cov_matrix_to_params
from estimagic.optimization.utilities import cov_matrix_to_sdcorr_params
from estimagic.optimization.utilities import cov_params_to_matrix
from estimagic.optimization.utilities import number_of_triangular_elements_to_dimension
from estimagic.optimization.utilities import sdcorr_params_to_matrix


def reparametrize_to_internal(params, constraints):
    """Convert a params DataFrame to an internal_params DataFrame.

    The internal params df is shorter because it does not contain fixed parameters.
    Moreover, it contains a reparametrized 'value' column that can be used to construct
    a parameter vector that satisfies all constraints. It also has adjusted lower and
    upper bounds.

    Args:
        params (DataFrame): A non-internal parameter DataFrame. See :ref:`params_df`.
        constraints (list): See :ref:`constraints`.

    Returns:
        internal (DataFrame): See :ref:`params_df`.

    """
    internal = params.copy()
    for constr in constraints:
        params_subset = params.loc[constr["index"]]
        if constr["type"] in ["covariance", "sdcorr"]:
            internal.update(
                _covariance_to_internal(params_subset, constr["case"], constr["type"])
            )
        elif constr["type"] == "sum":
            internal.update(_sum_to_internal(params_subset, constr["value"]))
        elif constr["type"] == "probability":
            internal.update(_probability_to_internal(params_subset))
        elif constr["type"] == "increasing":
            internal.update(_increasing_to_internal(params_subset))
        elif constr["type"] == "equality":
            internal.update(_equality_to_internal(params_subset))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        actually_fixed = internal["lower"] == internal["upper"]
        internal.loc[actually_fixed, "__fixed__"] = True

    internal = internal.query("~__fixed__")

    assert (internal["value"] >= internal["lower"]).all(), "Invalid lower bound."
    assert (internal["value"] <= internal["upper"]).all(), "Invalid upper bound."

    return internal


def reparametrize_from_internal(internal_params, constraints, original_params):
    """Convert an internal_params DataFrame to a Series with valid parameters.

    The parameter values are constructed from the 'value' column of internal_params.
    The resulting Series has the same index as the non-internal params DataFrame.

    Args:
        internal_params (DataFrame): internal parameter DataFrame. See :ref:`params_df`.
        constraints (list): see :ref:`constraints`.
        original_params (DataFrame): A non-internal parameter DataFrame. This is used to
            extract the original index and fixed values of parameters.

    Returns:
        params_sr (Series): See :ref:`params_df`.

    """
    reindexed = internal_params.reindex(original_params.index)
    params_sr = reindexed["value"].copy()
    fixed_index = params_sr[params_sr.isnull()].index

    # writing the fixed parameters back has to be done before all other constraints
    # are handled!
    params_sr.update(original_params.loc[fixed_index, "value"])

    for constr in constraints:
        params_subset = reindexed.loc[constr["index"]]
        if constr["type"] in ["covariance", "sdcorr"]:
            params_sr.update(
                _covariance_from_internal(params_subset, constr["case"], constr["type"])
            )
        elif constr["type"] == "sum":
            params_sr.update(_sum_from_internal(params_subset, constr["value"]))
        elif constr["type"] == "probability":
            params_sr.update(_probability_from_internal(params_subset))
        elif constr["type"] == "increasing":
            params_sr.update(_increasing_from_internal(params_subset))
        elif constr["type"] == "equality":
            params_sr.update(_equality_from_internal(params_subset))

    return params_sr


def _covariance_to_internal(params_subset, case, type_):
    """Reparametrize parameters that describe a covariance matrix to internal.

    If `type_` == 'covariance', the parameters in params_subset are assumed to be the
    lower triangular elements of a covariance matrix.

    If `type_` == 'sdcorr', the first *dim* parameters in params_subset are assumed to
    variances and the remaining parameters are assumed to be correlations.

    What has to be done depends on the case:
        - 'all_fixed': nothing has to be done
        - 'uncorrelated': bounds of diagonal elements are set to zero unless already
            stricter
        - 'all_free': do a (lower triangular) Cholesky reparametrization and restrict
            diagonal elements to be positive (see: https://tinyurl.com/y2n55cfb)

    Note that the cholesky reparametrization is not compatible with any other
    constraints on the involved parameters. Moreover, it requires the covariance matrix
    described by the start values to be positive definite as opposed to positive
    semi-definite.

    Args:
        params_subset (DataFrame): relevant subset of non-internal params.
        case (str): can take the values 'all_free', 'uncorrelated' or 'all_fixed'.

    Returns:
        res (DataFrame): copy of params_subset with adjusted 'value' and 'lower' columns

    """
    res = params_subset.copy()
    if type_ == "covariance":
        cov = cov_params_to_matrix(params_subset["value"].to_numpy())
    elif type_ == "sdcorr":
        cov = sdcorr_params_to_matrix(params_subset["value"].to_numpy())
    else:
        raise ValueError("Invalid type_: {}".format(type_))

    dim = len(cov)

    e, v = np.linalg.eigh(cov)
    assert np.all(e > -1e-8), "Invalid covariance matrix."

    if case == "uncorrelated":

        res["lower"] = np.maximum(res["lower"], np.zeros(len(res)))
        assert (res["upper"] >= res["lower"]).all(), "Invalid upper bound for variance."
    elif case == "all_free":
        chol = np.linalg.cholesky(cov)
        chol_coeffs = chol[np.tril_indices(dim)]
        res["value"] = chol_coeffs

        if type_ == "covariance":
            lower_bound_helper = np.full((dim, dim), -np.inf)
            lower_bound_helper[np.diag_indices(dim)] = 0
            res["lower"] = lower_bound_helper[np.tril_indices(dim)]
            res["upper"] = np.inf
            res["__fixed__"] = False
        else:
            res.loc[res.index[:dim], "lower"] = 0

        for bound in ["lower", "upper"]:
            if np.isfinite(params_subset[bound]).any():
                warnings.warn(
                    "Bounds are ignored for covariance parameters.", UserWarning
                )
    return res


def _covariance_from_internal(params_subset, case, type_):
    """Reparametrize parameters that describe a covariance matrix from internal.

    If case == 'all_free', undo the cholesky reparametrization. Otherwise, do nothing.

    Args:
        params_subset (DataFrame): relevant subset of internal_params.
        case (str): can take the values 'all_free', 'uncorrelated' or 'all_fixed'.

    Returns:
        res (Series): Series with lower triangular elements of a covariance matrix

    """
    res = params_subset.copy(deep=True)
    if case == "all_free":
        dim = number_of_triangular_elements_to_dimension(len(params_subset))
        helper = np.zeros((dim, dim))
        helper[np.tril_indices(dim)] = params_subset["value"].to_numpy()
        cov = helper.dot(helper.T)

        if type_ == "covariance":
            res["value"] = cov_matrix_to_params(cov)
        elif type_ == "sdcorr":
            res["value"] = cov_matrix_to_sdcorr_params(cov)
        else:
            raise ValueError("Invalid type_: {}".format(type_))
    elif case in ["all_fixed", "uncorrelated"]:
        pass
    else:
        raise ValueError("Invalid case: {}".format(case))
    return res["value"]


def _increasing_to_internal(params_subset):
    """Reparametrize increasing parameters to internal.

    Replace all but the first parameter by the difference to the previous one and
    set their lower bound to 0.

    Args:
        params_subset (DataFrame): relevant subset of non-internal params.

    Returns:
        res (DataFrame): copy of params_subset with adjusted 'value' and 'lower' columns

    """
    old_vals = params_subset["value"].to_numpy()
    new_vals = old_vals.copy()
    new_vals[1:] -= old_vals[:-1]
    res = params_subset.copy()
    res["value"] = new_vals

    res["__fixed__"] = False
    res["lower"] = [-np.inf] + [0] * (len(params_subset) - 1)
    res["upper"] = np.inf

    if params_subset["__fixed__"].any():
        warnings.warn("Ordered parameters were unfixed.", UserWarning)

    for bound in ["lower", "upper"]:
        if np.isfinite(params_subset[bound]).any():
            warnings.warn("Bounds are ignored for ordered parameters.", UserWarning)

    return res


def _increasing_from_internal(params_subset):
    """Reparametrize increasing parameters from internal.

    Replace the parameters by their cumulative sum.

    Args:
        params_subset (DataFrame): relevant subset of internal_params.

    Returns:
        res (Series): Series with increasing parameters.

    """
    res = params_subset.copy()
    res["value"] = params_subset["value"].cumsum()
    return res["value"]


def _sum_to_internal(params_subset, value):
    """Reparametrize sum constrained parameters to internal.

    fix the last parameter in params_subset.

    Args:
        params_subset (DataFrame): relevant subset of non-internal params.

    Returns:
        res (DataFrame): copy of params_subset with adjusted 'fixed' column

    """

    free = params_subset.query("lower == -inf & upper == inf & __fixed__ == False")
    last = params_subset.index[-1]

    assert (
        last in free.index
    ), "The last sum constrained parameter cannot have bounds nor be fixed."

    res = params_subset.copy()
    res.loc[last, "__fixed__"] = True
    return res


def _sum_from_internal(params_subset, value):
    """Reparametrize sum constrained parameters from internal.

    Replace the last parameter by *value* - the sum of all other parameters.

    Args:
        params_subset (DataFrame): relevant subset of internal_params.

    Returns:
        res (Series): parameters that sum to *value*

    """
    res = params_subset.copy()
    last = params_subset.index[-1]
    all_others = params_subset.index[:-1]
    res.loc[last, "value"] = value - params_subset.loc[all_others, "value"].sum()
    return res["value"]


def _probability_to_internal(params_subset):
    """Reparametrize probability constrained parameters to internal.

    fix the last parameter in params_subset,  divide all parameters by the last one
    and set all lower bounds to 0.

    Args:
        params_subset (DataFrame): relevant subset of non-internal params.

    Returns:
        res (DataFrame): copy of params_subset with adjusted 'fixed' and 'value'
            and 'lower' columns.

    """
    res = params_subset.copy()
    assert (
        params_subset["lower"].isin([-np.inf, 0]).all()
    ), "Lower bound has to be 0 or -inf for probability constrained parameters."

    assert (
        params_subset["upper"].isin([np.inf, 1]).all()
    ), "Upper bound has to be 1 or inf for probability constrained parameters."

    assert not params_subset[
        "__fixed__"
    ].any(), "Probability constrained parameters cannot be fixed."

    res["lower"] = 0
    res["upper"] = np.inf
    last = params_subset.index[-1]
    res.loc[last, "__fixed__"] = True
    res["value"] /= res.loc[last, "value"]
    return res


def _probability_from_internal(params_subset):
    """Reparametrize probability constrained parameters from internal.

    Replace the last parameter by 1 and divide by the sum of all parameters.

    Args:
        params_subset (DataFrame): relevant subset of internal_params.

    Returns:
        res (Series): parameters that sum to 1 and are between 0 and 1.

    """
    last = params_subset.index[-1]
    res = params_subset.copy()
    res.loc[last, "value"] = 1
    res["value"] /= res["value"].sum()
    return res["value"]


def _equality_to_internal(params_subset):
    """Reparametrize equality constrained parameters to internal.

    fix all but the first parameter in params_subset

    Args:
        params_subset (DataFrame): relevant subset of non-internal params.

    Returns:
        res (DataFrame): copy of params_subset with adjusted 'fixed' column

    """
    res = params_subset.copy()
    first = params_subset.index[0]
    all_others = params_subset.index[1:]
    res.loc[first, "__fixed__"] = params_subset["__fixed__"].any()
    res.loc[all_others, "__fixed__"] = True
    res["lower"] = params_subset["lower"].max()
    res["upper"] = params_subset["upper"].min()
    assert (
        res["lower"] <= res["upper"]
    ).all(), "Invalid bounds for equality constrained parameters."
    assert len(params_subset["value"].unique()) == 1, "Equality constraint is violated."
    return res


def _equality_from_internal(params_subset):
    """Reparametrize equality constrained parameters from internal.

    Replace the previously fixed parameters by the first parameter

    Args:
        params_subset (DataFrame): relevant subset of internal_params.

    Returns:
        res (Series): parameters that obey the equality constraint.

    """
    res = params_subset.copy()
    first = params_subset.index[0]
    all_others = params_subset.index[1:]
    res.loc[all_others, "value"] = res.loc[first, "value"]
    return res["value"]
