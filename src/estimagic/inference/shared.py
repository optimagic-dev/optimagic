import functools
from collections.abc import Callable

import numpy as np
import pandas as pd
import scipy
from estimagic.decorators import numpy_interface
from estimagic.differentiation.derivatives import first_derivative
from estimagic.parameters.parameter_conversion import get_internal_bounds
from estimagic.parameters.parameter_conversion import get_reparametrize_functions
from estimagic.parameters.process_constraints import process_constraints


def transform_covariance(
    params,
    internal_cov,
    constraints,
    n_samples,
    bounds_handling,
):
    """Transform the internal covariance matrix to an external one, given constraints.

    Args:
        params (pd.DataFrame): DataFrame where the "value" column contains estimated
            parameters of a likelihood model. See :ref:`params` for details.
        internal_cov (np.ndarray or pandas.DataFrame) with a covariance matrix of the
            internal parameter vector. For background information about internal and
            external params see :ref:`implementation_of_constraints`.
        constraints (list): List with constraint dictionaries.
            See .. _link: ../../docs/source/how_to_guides/how_to_use_constraints.ipynb
        n_samples (int): Number of samples used to transform the covariance matrix of
            the internal parameter vector into the covariance matrix of the external
            parameters.
        bounds_handling (str): One of "clip", "raise", "ignore". Determines how bounds
            are handled. If "clip", confidence intervals are clipped at the bounds.
            Standard errors are only adjusted if a sampling step is necessary due to
            additional constraints. If "raise" and any lower or upper bound is binding,
            we raise an error. If "ignore", boundary problems are simply ignored.

    Returns:
        pd.DataFrame: Quadratic DataFrame containing the covariance matrix of the free
            parameters. If parameters were fixed (explicitly or by other constraints),
            the index is a subset of params.index. The columns are the same as the
            index.

    """
    processed_constraints, processed_params = process_constraints(constraints, params)
    free_index = processed_params.query("_internal_free").index

    if isinstance(internal_cov, pd.DataFrame):
        internal_cov = internal_cov.to_numpy()

    if processed_constraints:
        _to_internal, _from_internal = get_reparametrize_functions(
            params=params, constraints=constraints
        )

        free = processed_params.loc[free_index]
        is_free = processed_params["_internal_free"].to_numpy()
        lower_bounds = free["_internal_lower"]
        upper_bounds = free["_internal_upper"]

        internal_mean = _to_internal(params)

        sample = np.random.multivariate_normal(
            mean=internal_mean,
            cov=internal_cov,
            size=n_samples,
        )
        transformed_free = []
        for params_vec in sample:
            if bounds_handling == "clip":
                params_vec = np.clip(params_vec, a_min=lower_bounds, a_max=upper_bounds)
            elif bounds_handling == "raise":
                if (params_vec < lower_bounds).any() or (
                    params_vec > upper_bounds
                ).any():
                    raise ValueError()

            transformed = _from_internal(internal=params_vec)
            transformed_free.append(transformed[is_free])

        free_cov = np.cov(
            np.array(transformed_free),
            rowvar=False,
        )

    else:
        free_cov = internal_cov

    res = pd.DataFrame(data=free_cov, columns=free_index, index=free_index)
    return res


def calculate_inference_quantities(params, free_cov, ci_level):
    """Add standard errors, pvalues and confidence intervals to params.

    Args
        params (pd.DataFrame): See :ref:`params`.
        free_cov (pd.DataFrame): Quadratic DataFrame containing the covariance matrix
            of the free parameters. If parameters were fixed (explicitly or by other
            constraints) the index is a subset of params.index. The columns are the same
            as the index.
        ci_level (float): Confidence level for the calculation of confidence intervals.

    Returns:
        pd.DataFrame: DataFrame with same index as params, containing the columns
            "value", "standard_error", "pvalue", "ci_lower" and "ci_upper".
            Parameters that do not have a standard error (e.g. because they were fixed
            during estimation) contain NaNs in all but the "value" column. The value
            column is only reproduced for convenience.

    """
    free = pd.DataFrame(index=free_cov.index)
    free["value"] = params.loc[free.index, "value"]
    free["standard_error"] = np.sqrt(np.diag(free_cov))
    tvalues = free["value"] / free["standard_error"]
    free["p_value"] = 1.96 * scipy.stats.norm.sf(np.abs(tvalues))

    alpha = 1 - ci_level
    scale = scipy.stats.norm.ppf(1 - alpha / 2)
    free["ci_lower"] = free["value"] - scale * free["standard_error"]
    free["ci_upper"] = free["value"] + scale * free["standard_error"]

    free["stars"] = pd.cut(
        free["p_value"], bins=[-1, 0.01, 0.05, 0.1, 2], labels=["***", "**", "*", ""]
    )

    res = free.reindex(params.index)
    res["value"] = params["value"]
    return res


def get_internal_first_derivative(
    func, params, constraints=None, func_kwargs=None, numdiff_options=None
):
    """Get the first_derivative of func with respect to internal parameters.

    If there are no constraints, we simply call the first_derivative function.

    Args:
        func (callable): Function to take the derivative of.
        params (pandas.DataFrame): Data frame with external parameters. See
            :ref:`params`.
        constraints (list): Constraints that define how to convert between internal
            and external parameters.
        func_kwargs (dict): Additional keyword arguments for func.
        numdiff_options (dict): Additional options for first_derivative.

    Returns:
        dict: See ``first_derivative`` for details. The only difference is that the
            the "derivative" entry is always a numpy array instead of a DataFrame

    """
    numdiff_options = {} if numdiff_options is None else numdiff_options
    func_kwargs = {} if func_kwargs is None else func_kwargs
    _func = functools.partial(func, **func_kwargs)

    if constraints is None:
        out = first_derivative(
            func=_func,
            params=params,
            **numdiff_options,
        )
        out["has_transforming_constraints"] = False
    else:

        lower_bounds, upper_bounds = get_internal_bounds(params, constraints)

        _internal_func = numpy_interface(
            func=_func, params=params, constraints=constraints
        )

        _to_internal, _ = get_reparametrize_functions(params, constraints)

        _x = _to_internal(params)

        out = first_derivative(
            _internal_func,
            _x,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            **numdiff_options,
        )

        if isinstance(out["derivative"], (pd.DataFrame, pd.Series)):
            out["derivative"] = out["derivative"].to_numpy()

    return out


def process_pandas_arguments(**kwargs):
    """Convert pandas objects to arrays and extract names of moments and parameters.

    This works for any number of keyword arguments. The result is a tuple containing
    numpy arrays in same order as the keyword arguments and a dictionary with
    the separated index objects as last entry. This dictionary contains the entries
    "moments" and "params" for the identified moment names and parameter names.

    The keyword arguments "jac", "hess", "weights" and "moments_cov" are used to extract
    the names. Other keyword arguments are simply converted to numpy arrays.

    """
    param_name_candidates = {}
    moment_name_candidates = {}

    if "jac" in kwargs:
        jac = kwargs["jac"]
        if isinstance(jac, pd.DataFrame):
            param_name_candidates["jac"] = jac.columns
            moment_name_candidates["jac"] = jac.index

    if "hess" in kwargs:
        hess = kwargs["hess"]
        if isinstance(hess, pd.DataFrame):
            param_name_candidates["hess"] = hess.index

    if "weights" in kwargs:
        weights = kwargs["weights"]
        if isinstance(weights, pd.DataFrame):
            moment_name_candidates["weights"] = weights.index

    if "moments_cov" in kwargs:
        moments_cov = kwargs["moments_cov"]
        if isinstance(moments_cov, pd.DataFrame):
            moment_name_candidates["moments_cov"] = moments_cov.index

    names = {}
    if param_name_candidates:
        _check_names_coincide(param_name_candidates)
        names["params"] = list(param_name_candidates.values())[0]
    if moment_name_candidates:
        _check_names_coincide(moment_name_candidates)
        names["moments"] = list(moment_name_candidates.values())[0]

    # order of outputs is same as order of inputs; names are last.
    out_list = [_to_numpy(val, name=key) for key, val in kwargs.items()] + [names]
    return tuple(out_list)


def _to_numpy(df_or_array, name):
    if isinstance(df_or_array, pd.DataFrame):
        arr = df_or_array.to_numpy()
    elif isinstance(df_or_array, np.ndarray):
        arr = df_or_array
    else:
        raise ValueError(
            f"{name} must be a DataFrame or numpy array, not {type(df_or_array)}."
        )
    return arr


def _check_names_coincide(name_dict):
    if len(name_dict) >= 2:
        first_key = list(name_dict)[0]
        first_names = name_dict[first_key]

        for key, names in name_dict.items():
            if not first_names.equals(names):
                msg = f"Ambiguous parameter or moment names from {first_key} and {key}."
                raise ValueError(msg)


def get_derivative_case(derivative):
    """Determine which kind of derivative should be used."""
    if isinstance(derivative, (pd.DataFrame, np.ndarray)):
        case = "pre-calculated"
    elif isinstance(derivative, Callable):
        case = "closed-form"
    elif derivative is False:
        case = "skip"
    else:
        case = "numerical"
    return case


def check_is_optimized_and_derivative_case(is_minimized, derivative_case):
    if (not is_minimized) and derivative_case == "pre-calculated":
        raise ValueError(
            "Providing a pre-calculated derivative is only possible if the "
            "optimization was done outside of the estimate_function, i.e. if "
            "optimize_options=False."
        )
