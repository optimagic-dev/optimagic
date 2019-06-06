"""Handle constraints by bounds and reparametrizations."""
import warnings

import numpy as np
import pandas as pd

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
        constraints (list): See :ref:`constraints`. It is assumed that the constraints
            are already processed and sorted.

    Returns:
        internal (DataFrame): See :ref:`params_df`.

    """
    internal = params.copy()

    internal["_fixed"] = False
    fixed_constraints = [c for c in constraints if c["type"] == "fixed"]
    for constr in fixed_constraints:
        internal.loc[constr["index"], "_fixed"] = True
        internal.loc[constr["index"], "value"] = constr["value"]

    for constr in constraints:
        params_subset = internal.loc[constr["index"]]
        if constr["type"] == "equality":
            internal.update(_equality_to_internal(internal.loc[constr["index"]]))
        elif constr["type"] in ["covariance", "sdcorr"]:
            internal.update(
                _covariance_to_internal(params_subset, constr["case"], constr["type"])
            )
        elif constr["type"] == "sum":
            internal.update(_sum_to_internal(params_subset, constr["value"]))
        elif constr["type"] == "probability":
            internal.update(_probability_to_internal(params_subset))
        elif constr["type"] == "increasing":
            internal.update(_increasing_to_internal(params_subset))
        elif constr["type"] == "fixed":
            pass
        else:
            raise ValueError("Invalid constraint type: {}".format(constr["type"]))

    internal["_fixed"] = internal["_fixed"].astype(bool)
    assert internal["lower"].lt(internal["upper"]).all(), "lower must be < upper."
    internal = internal.loc[~(internal["_fixed"])].copy(deep=True)
    internal.drop(["_fixed"], axis=1, inplace=True)

    assert internal["value"].ge(internal["lower"]).all(), "Invalid lower bound."
    assert internal["value"].le(internal["upper"]).all(), "Invalid upper bound."

    return internal


def make_start_params_helpers(params_index, extended_constraints):
    """Helper DataFrames to generate start params.

    Construct a default params DataFrame and split it into free and parameters and
    parameters that are fixed explicitly or implicitly through equality constraints.

    The free parameters can be exposed to a user to generate custom start parameters
    in a complex model. The fixed part can then be used to transform the user provided
    start parameters into a full params_df.

    Args:
        params_index (DataFrame): The index of a non-internal parameter DataFrame.
            See :ref:`params_df`.
        extended_constraints (list): A list of constraints where in addition to what
            is described in :ref:`constraints` all fixed parameters also contain a
            'value' entry.

    Returns:
        free (DataFrame): free parameters
        fixed (DataFrame): parameters that are fixed because of explicit fixes
            or equality constraints.

    """
    params = pd.DataFrame(index=params_index)
    params["lower"] = -np.inf
    params["upper"] = np.inf
    params["value"] = np.nan

    # handle explicit fixes
    params["_fixed"] = False
    fixed_constraints = [c for c in extended_constraints if c["type"] == "fixed"]
    for constr in fixed_constraints:
        params.loc[constr["index"], "_fixed"] = True
        params.loc[constr["index"], "value"] = constr["value"]

    equality_constraints = [c for c in extended_constraints if c["type"] == "equality"]
    for constr in equality_constraints:
        params.update(_equality_to_internal(params.loc[constr["index"]]))

    free = params.query("~_fixed").drop("_fixed", axis=1)
    fixed = params.query("_fixed").drop("_fixed", axis=1)
    return free, fixed


def get_start_params_from_helpers(free, fixed, constraints, params_index):
    """Construct a params_df from helper DataFrames.

    Args:
        free (DataFrame): free parameters
        fixed (DataFrame): parameters that are fixed explicitly or due to equality
            constraints.
        constraints (list): list of constraints
        params_index (DataFrame): The index of a non-internal parameter DataFrame.
            See :ref:`params_df`.

    Returns:
        params (DataFrame): see :ref:`params_df`.

    """
    equality_constraints = [c for c in constraints if c["type"] == "equality"]
    params = pd.concat([free, fixed], axis=0).loc[params_index]
    for constr in equality_constraints:
        params_subset = params.loc[constr["index"]]
        values = params_subset["value"].unique()
        assert len(values) == 1, "Too many values."
        params.loc[constr["index"]] = values[0]
    return params


def reparametrize_from_internal(internal_params, constraints, original_params):
    """Convert an internal_params DataFrame to a Series with valid parameters.

    The parameter values are constructed from the 'value' column of internal_params.
    The resulting Series has the same index as the non-internal params DataFrame.

    Args:
        internal_params (DataFrame): internal parameter DataFrame. See :ref:`params_df`.
        constraints (list): see :ref:`constraints`. It is assumed that the constraints
            are already processed.
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
        if constr["type"] == "equality":
            params_subset = reindexed.loc[constr["index"]]
            reindexed.update(_equality_from_internal(params_subset))
            params_sr.update(_equality_from_internal(params_subset))

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
            res["_fixed"] = False
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

    res["_fixed"] = False
    res["lower"] = [-np.inf] + [0] * (len(params_subset) - 1)
    res["upper"] = np.inf

    if params_subset["_fixed"].any():
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

    free = params_subset.query("lower == -inf & upper == inf & _fixed == False")
    last = params_subset.index[-1]

    assert (
        last in free.index
    ), "The last sum constrained parameter cannot have bounds nor be fixed."

    res = params_subset.copy()
    res.loc[last, "_fixed"] = True
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
        "_fixed"
    ].any(), "Probability constrained parameters cannot be fixed."

    res["lower"] = 0
    res["upper"] = np.inf
    last = params_subset.index[-1]
    res.loc[last, "_fixed"] = True
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
    if params_subset["_fixed"].any():
        res.loc[first, "_fixed"] = True
    res.loc[all_others, "_fixed"] = True
    res["lower"] = params_subset["lower"].max()
    res["upper"] = params_subset["upper"].min()
    assert len(params_subset["value"].unique()) == 1, "Equality constraint is violated."
    assert res["_fixed"].dtype == bool
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
