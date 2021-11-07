import functools
import re
from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd
from estimagic import batch_evaluators
from estimagic.config import DEFAULT_N_CORES
from estimagic.differentiation import finite_differences
from estimagic.differentiation.generate_steps import generate_steps
from estimagic.differentiation.richardson_extrapolation import richardson_extrapolation
from estimagic.utilities import namedtuple_from_kwargs


def first_derivative(
    func,
    params,
    func_kwargs=None,
    method="central",
    n_steps=1,
    base_steps=None,
    scaling_factor=1,
    lower_bounds=None,
    upper_bounds=None,
    step_ratio=2,
    min_steps=None,
    f0=None,
    n_cores=DEFAULT_N_CORES,
    error_handling="continue",
    batch_evaluator="joblib",
    return_func_value=False,
    return_info=True,
    key=None,
):
    """Evaluate first derivative of func at params according to method and step options.

    Internally, the function is converted such that it maps from a 1d array to a 1d
    array. Then the Jacobian of that function is calculated. The resulting derivative
    estimate is always a :class:`numpy.ndarray`.

    The parameters and the function output can be pandas objects (Series or DataFrames
    with value column). In that case the output of first_derivative is also a pandas
    object and with appropriate index and columns.

    Detailed description of all options that influence the step size as well as an
    explanation of how steps are adjusted to bounds in case of a conflict,
    see :func:`~estimagic.differentiation.generate_steps.generate_steps`.

    Args:
        func (callable): Function of which the derivative is calculated.
        params (numpy.ndarray, pandas.Series or pandas.DataFrame): 1d numpy array or
            :class:`pandas.DataFrame` with parameters at which the derivative is
            calculated. If it is a DataFrame, it can contain the columns "lower_bound"
            and "upper_bound" for bounds. See :ref:`params`.
        func_kwargs (dict): Additional keyword arguments for func, optional.
        method (str): One of ["central", "forward", "backward"], default "central".
        n_steps (int): Number of steps needed. For central methods, this is
            the number of steps per direction. It is 1 if no Richardson extrapolation
            is used.
        base_steps (numpy.ndarray, optional): 1d array of the same length as pasams.
            base_steps * scaling_factor is the absolute value of the first (and possibly
            only) step used in the finite differences approximation of the derivative.
            If base_steps * scaling_factor conflicts with bounds, the actual steps will
            be adjusted. If base_steps is not provided, it will be determined according
            to a rule of thumb as long as this does not conflict with min_steps.
        scaling_factor (numpy.ndarray or float): Scaling factor which is applied to
            base_steps. If it is an numpy.ndarray, it needs to be as long as params.
            scaling_factor is useful if you want to increase or decrease the base_step
            relative to the rule-of-thumb or user provided base_step, for example to
            benchmark the effect of the step size. Default 1.
        lower_bounds (numpy.ndarray): 1d array with lower bounds for each parameter. If
            params is a DataFrame and has the columns "lower_bound", this will be taken
            as lower_bounds if now lower_bounds have been provided explicitly.
        upper_bounds (numpy.ndarray): 1d array with upper bounds for each parameter. If
            params is a DataFrame and has the columns "upper_bound", this will be taken
            as upper_bounds if no upper_bounds have been provided explicitly.
        step_ratio (float, numpy.array): Ratio between two consecutive Richardson
            extrapolation steps in the same direction. default 2.0. Has to be larger
            than one. The step ratio is only used if n_steps > 1.
        min_steps (numpy.ndarray): Minimal possible step sizes that can be chosen to
            accommodate bounds. Must have same length as params. By default min_steps is
            equal to base_steps, i.e step size is not decreased beyond what is optimal
            according to the rule of thumb.
        f0 (numpy.ndarray): 1d numpy array with func(x), optional.
        n_cores (int): Number of processes used to parallelize the function
            evaluations. Default 1.
        error_handling (str): One of "continue" (catch errors and continue to calculate
            derivative estimates. In this case, some derivative estimates can be
            missing but no errors are raised), "raise" (catch errors and continue
            to calculate derivative estimates at fist but raise an error if all
            evaluations for one parameter failed) and "raise_strict" (raise an error
            as soon as a function evaluation fails).
        batch_evaluator (str or callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the estimagic batch_evaluators.
        return_func_value (bool): If True, return function value at params, stored in
            output dict under "func_value". Default False. This is useful when using
            first_derivative during optimization.
        return_info (bool): If True, return additional information on function
            evaluations and internal derivative candidates, stored in output dict under
            "func_evals" and "derivative_candidates". Derivative candidates are only
            returned if n_steps > 1. Default True.
        key (str): If func returns a dictionary, take the derivative of
            func(params)[key].

    Returns:
        result (dict): Result dictionary with keys:
            - "derivative" (numpy.ndarray, pandas.Series or pandas.DataFrame): The
                estimated first derivative of func at params. The shape of the output
                depends on the dimension of params and func(params):

                - f: R -> R leads to shape (1,), usually called derivative
                - f: R^m -> R leads to shape (m, ), usually called Gradient
                - f: R -> R^n leads to shape (n, 1), usually called Jacobian
                - f: R^m -> R^n leads to shape (n, m), usually called Jacobian

            - "func_value" (numpy.ndarray, pandas.Series or pandas.DataFrame): Function
                value at params, returned if return_func_value is True.

            - "func_evals" (pandas.DataFrame): Function evaluations produced by internal
                derivative method, returned if return_info is True.

            - "derivative_candidates" (pandas.DataFrame): Derivative candidates from
                Richardson extrapolation, returned if return_info is True and n_steps >
                1.

    """
    lower_bounds, upper_bounds = _process_bounds(lower_bounds, upper_bounds, params)

    # handle keyword arguments
    func_kwargs = {} if func_kwargs is None else func_kwargs
    partialed_func = functools.partial(func, **func_kwargs)

    # convert params to numpy, but keep label information
    params_index = (
        params.index if isinstance(params, (pd.DataFrame, pd.Series)) else None
    )

    x = params["value"].to_numpy() if isinstance(params, pd.DataFrame) else params
    x = np.atleast_1d(x).astype(float)

    if np.isnan(x).any():
        raise ValueError("The parameter vector must not contain NaNs.")

    # generate the step array
    steps = generate_steps(
        x=x,
        method=method,
        n_steps=n_steps,
        target="first_derivative",
        base_steps=base_steps,
        scaling_factor=scaling_factor,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        step_ratio=step_ratio,
        min_steps=min_steps,
    )

    # generate parameter vectors at which func has to be evaluated as numpy arrays
    evaluation_points = []
    for step_arr in steps:
        for i, j in product(range(n_steps), range(len(x))):
            if np.isnan(step_arr[i, j]):
                evaluation_points.append(np.nan)
            else:
                point = x.copy()
                point[j] += step_arr[i, j]
                evaluation_points.append(point)

    # convert the numpy arrays to whatever is needed by func
    evaluation_points = _convert_evaluation_points_to_original(
        evaluation_points, params
    )

    # we always evaluate f0, so we can fall back to one-sided derivatives if
    # two-sided derivatives fail. The extra cost is negligible in most cases.
    if f0 is None:
        evaluation_points.append(params)

    # do the function evaluations, including error handling
    batch_error_handling = "raise" if error_handling == "raise_strict" else "continue"
    raw_evals = _nan_skipping_batch_evaluator(
        func=partialed_func,
        arguments=evaluation_points,
        n_cores=n_cores,
        error_handling=batch_error_handling,
        batch_evaluator=batch_evaluator,
    )

    # extract information on exceptions that occurred during function evaluations
    exc_info = "\n\n".join([val for val in raw_evals if isinstance(val, str)])
    raw_evals = [val if not isinstance(val, str) else np.nan for val in raw_evals]

    # store full function value at params as func_value and a processed version of it
    # that we need to calculate derivatives as f0
    if f0 is None:
        f0 = raw_evals[-1]
        raw_evals = raw_evals[:-1]
    func_value = f0
    f0 = f0[key] if isinstance(f0, dict) else f0
    f_was_scalar = np.isscalar(f0)
    out_index = f0.index if isinstance(f0, pd.Series) else None
    f0 = np.atleast_1d(f0)

    # convert the raw evaluations to numpy arrays
    raw_evals = _convert_evals_to_numpy(raw_evals, key)

    # apply finite difference formulae
    evals = np.array(raw_evals).reshape(2, n_steps, len(x), -1)
    evals = np.transpose(evals, axes=(0, 1, 3, 2))
    evals = namedtuple_from_kwargs(pos=evals[0], neg=evals[1])

    jac_candidates = {}
    for m in ["forward", "backward", "central"]:
        jac_candidates[m] = finite_differences.jacobian(evals, steps, f0, m)

    # get the best derivative estimate out of all derivative estimates that could be
    # calculated, given the function evaluations.
    orders = {
        "central": ["central", "forward", "backward"],
        "forward": ["forward", "backward"],
        "backward": ["backward", "forward"],
    }

    if n_steps == 1:
        jac = _consolidate_one_step_derivatives(jac_candidates, orders[method])
        updated_candidates = None
    else:
        richardson_candidates = _compute_richardson_candidates(
            jac_candidates, steps, n_steps
        )
        jac, updated_candidates = _consolidate_extrapolated(richardson_candidates)

    # raise error if necessary
    if error_handling in ("raise", "raise_strict") and np.isnan(jac).any():
        raise Exception(exc_info)

    # results processing
    derivative = jac.flatten() if f_was_scalar else jac
    derivative = _add_index_to_derivative(derivative, params_index, out_index)

    result = {"derivative": derivative}
    if return_func_value:
        result["func_value"] = func_value

    info = _collect_additional_info(return_info, steps, evals, updated_candidates)
    result = {**result, **info}
    return result


def _process_bounds(lower_bounds, upper_bounds, params):
    lower_bounds = np.atleast_1d(lower_bounds) if lower_bounds is not None else None
    upper_bounds = np.atleast_1d(upper_bounds) if upper_bounds is not None else None
    if isinstance(params, pd.DataFrame):
        if lower_bounds is None and "lower_bound" in params.columns:
            lower_bounds = params["lower_bound"].to_numpy()
        if upper_bounds is None and "upper_bound" in params.columns:
            upper_bounds = params["upper_bound"].to_numpy()
    return lower_bounds, upper_bounds


def _convert_evaluation_points_to_original(evaluation_points, params):
    if np.isscalar(params):
        res = [p[0] if isinstance(p, np.ndarray) else p for p in evaluation_points]
    elif isinstance(params, pd.DataFrame):
        res = []
        for point in evaluation_points:
            if isinstance(point, np.ndarray):
                pandas_point = params.copy(deep=True)
                pandas_point["value"] = point
                res.append(pandas_point)
            else:
                res.append(point)
    elif isinstance(params, pd.Series):
        res = [
            pd.Series(p, index=params.index) if isinstance(p, np.ndarray) else p
            for p in evaluation_points
        ]
    else:
        res = evaluation_points
    return res


def _convert_evaluation_data_to_frame(steps, evals):
    """Convert evaluation data to (tidy) data frame.

    Args:
        steps (namedtuple): Namedtuple with field names pos and neg. Is generated by
            :func:`~estimagic.differentiation.generate_steps.generate_steps`.
        evals (namedtuple): Namedtuple with field names pos and neg. Contains function
            evaluation corresponding to steps.

    Returns:
        df (pandas.DataFrame): Tidy data frame with index (sign, step_number, dim_x
            dim_f), where sign corresponds to pos or neg in steps and evals, step_number
            indexes the step, dim_x is the dimension of the input vector and dim_f is
            the dimension of the function output. The data is given by the two columns
            step and eval. The data frame has 2 * n_steps * dim_x * dim_f rows.

    """
    n_steps, dim_f, dim_x = evals.pos.shape
    params_index = range(dim_x)

    dfs = []
    for direction, step_arr, eval_arr in zip((1, -1), steps, evals):
        df_steps = (
            pd.DataFrame(step_arr, columns=params_index)
            .reset_index()
            .rename(columns={"index": "step_number"})
            .melt(id_vars="step_number", var_name="dim_x", value_name="step")
            .sort_values("step_number")
            .reset_index(drop=True)
            .apply(lambda col: col.abs() if col.name == "step" else col)
        )
        eval_arr = np.transpose(eval_arr, (0, 2, 1)).reshape(-1, dim_f)
        df_evals = (
            pd.concat((df_steps, pd.DataFrame(eval_arr)), axis=1)
            .melt(
                id_vars=["step_number", "dim_x", "step"],
                var_name="dim_f",
                value_name="eval",
            )
            .assign(**{"sign": direction})
            .set_index(["sign", "step_number", "dim_x", "dim_f"])
            .sort_index()
        )
        dfs.append(df_evals)
    df = pd.concat(dfs).astype({"step": float, "eval": float})
    return df


def _convert_richardson_candidates_to_frame(jac, err):
    """Convert (richardson) jacobian candidates and errors to pandas data frame.

    Args:
        jac (dict): Dict with richardson jacobian candidates.
        err (dict): Dict with errors corresponding to richardson jacobian candidates.

    Returns:
        df (pandas.DataFrame): Frame with column "der" and "err" and index ["method",
            "num_term", "dim_x", "dim_f"] with respective meaning: type of method used,
            e.g. central or foward; kind of value, e.g. derivative or error.

    """
    dim_f, dim_x = jac["forward1"].shape
    dfs = []
    for key, value in jac.items():
        method, num_term = _split_into_str_and_int(key)
        df = pd.DataFrame(value.T, columns=range(dim_f))
        df = df.assign(**{"dim_x": range(dim_x)})
        df = df.melt(id_vars="dim_x", var_name="dim_f", value_name="der")
        df = df.assign(
            **{"method": method, "num_term": num_term, "err": err[key].T.flatten()}
        )
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.set_index(["method", "num_term", "dim_x", "dim_f"])
    return df


def _convert_evals_to_numpy(raw_evals, key):
    """harmonize the output of the function evaluations.

    The raw_evals might contain dictionaries of which we only need one entry, scalar
    np.nan where we need arrays filled with np.nan or pandas objects. The processed
    evals only contain numpy arrays.

    """
    # get rid of dictionaries
    evals = [val[key] if isinstance(val, dict) else val for val in raw_evals]
    # get rid of pandas objects
    evals = [np.array(val) if isinstance(val, pd.Series) else val for val in evals]

    # find out the correct output shape
    try:
        array = next(x for x in evals if hasattr(x, "shape") or isinstance(x, dict))
        out_shape = array.shape
    except StopIteration:
        out_shape = "scalar"

    # convert to correct output shape
    if out_shape == "scalar":
        evals = [np.atleast_1d(val) for val in evals]
    else:
        for i in range(len(evals)):
            if isinstance(evals[i], float) and np.isnan(evals[i]):
                evals[i] = np.full(out_shape, np.nan)

    return evals


def _consolidate_one_step_derivatives(candidates, preference_order):
    """Replace missing derivative estimates of preferred method with others.

    Args:
        candidates (dict): Dictionary with derivative estimates from different methods.
        preference_order (list): Order on (a subset of) the keys in candidates. Earlier
        entries are preferred.

    Returns:
        consolidated (np.ndarray): Array of same shape as input derivative estimates.

    """
    preferred, others = preference_order[0], preference_order[1:]
    consolidated = candidates[preferred].copy()
    for other in others:
        consolidated = np.where(np.isnan(consolidated), candidates[other], consolidated)

    return consolidated.reshape(consolidated.shape[1:])


def _consolidate_extrapolated(candidates):
    """Get the best possible derivative estimate, given an error estimate.

    Going through ``candidates`` select the best derivative estimate element-wise using
    the estimated candidates, where best is defined as minimizing the error estimate
    from the Richardson extrapolation.

    See https://tinyurl.com/ubn3nv5 for corresponding code in numdifftools and
    https://tinyurl.com/snle7mb for an explanation of how errors of Richardson
    extrapolated derivative estimates can be estimated.

    Args:
        candidates (OrderedDict): Dictionary containing different derivative estimates
            and their error estimates.

    Returns:
        consolidated (np.ndarray): Array of same shape as input derivative estimates.
        candidate_der_dict (dict): Best derivative estimate given method.
        candidate_err_dict (dict): Errors corresponding to best derivatives given method

    """
    # first find minimum over steps for each method
    candidate_der_dict = {}
    candidate_err_dict = {}

    for key in candidates.keys():
        _der = candidates[key]["derivative"]
        _err = candidates[key]["error"]
        derivative, error = _select_minimizer_along_axis(_der, _err)
        candidate_der_dict[key] = derivative
        candidate_err_dict[key] = error

    # second find minimum over methods
    candidate_der = np.stack(list(candidate_der_dict.values()))
    candidate_err = np.stack(list(candidate_err_dict.values()))
    consolidated, _ = _select_minimizer_along_axis(candidate_der, candidate_err)

    updated_candidates = (candidate_der_dict, candidate_err_dict)
    return consolidated, updated_candidates


def _compute_richardson_candidates(jac_candidates, steps, n_steps):
    """Compute derivative candidates using Richardson extrapolation.

    Args:
        jac_candidates (dict): Dictionary with (traditional) derivative estimates from
            different methods.
        steps (namedtuple): Namedtuple with the field names pos and neg. Each field
            contains a numpy array of shape (n_steps, len(x)) with the steps in
            the corresponding direction. The steps are always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.
        n_steps (int): Number of steps needed. For central methods, this is
            the number of steps per direction. It is 1 if no Richardson extrapolation
            is used.

    Returns:
        richardson_candidates (OrderedDict): Dictionary with derivative estimates and
            error estimates from different methods.
            - Keys correspond to the method used, i.e. forward, backward or central
            differences and the number of terms used in the Richardson extrapolation.
            - Values represent the corresponding derivative estimate and error
            estimate, stored as np.ndarrays in a sub-dictionary under "derivative" and
            "error" respectively, with the first dimensions coinciding with that of an
            element of ``jac_candidates`` and depending on num_terms, possibly one
            further dimension.

    """
    richardson_candidates = OrderedDict()
    for method in ["forward", "backward", "central"]:
        for num_terms in range(1, n_steps):
            derivative, error = richardson_extrapolation(
                jac_candidates[method], steps, method, num_terms
            )
            richardson_candidates[method + str(num_terms)] = {
                "derivative": derivative,
                "error": error,
            }

    return richardson_candidates


def _select_minimizer_along_axis(derivative, errors):
    """Select best derivative estimates element wise.

    Select elements from ``derivative`` which correspond to minimum in ``errors`` along
    first axis.

    Args:
        derivative (np.ndarray): Derivative estimates from Richardson approximation.
            First axis (axis 0) denotes the potentially multiple estimates. Following
            dimensions represent the dimension of the derivative, i.e. for a classical
            gradient ``derivative`` has 2 dimensions, while for a classical jacobian
            ``derivative`` has 3 dimensions.
        errors (np.ndarray): Error estimates of ``derivative`` estimates. Has the same
            shape as ``derivative``.

    Returns:
        derivative_minimal (np.ndarray): Best derivate estimates chosen with respect
            to minimizing ``errors``. Note that the best values are selected
            element-wise. Has shape ``(derivative.shape[1], derivative.shape[2])``.

        error_minimal (np.ndarray): Minimal errors selected element-wise along axis
            0 of ``errors``.

    """
    if derivative.shape[0] == 1:
        jac_minimal = np.squeeze(derivative, axis=0)
        error_minimal = np.squeeze(errors, axis=0)
    else:
        minimizer = np.nanargmin(errors, axis=0)
        jac_minimal = np.take_along_axis(derivative, minimizer[np.newaxis, :], axis=0)
        jac_minimal = np.squeeze(jac_minimal, axis=0)
        error_minimal = np.nanmin(errors, axis=0)

    return jac_minimal, error_minimal


def _nan_skipping_batch_evaluator(
    func, arguments, n_cores, error_handling, batch_evaluator
):
    """Evaluate func at each entry in arguments, skipping np.nan entries.

    The function is only evaluated at inputs that are not a scalar np.nan.
    The outputs corresponding to skipped inputs as well as for inputs on which func
    returns np.nan are np.nan.

    Args:
        func (function): Python function that returns a numpy array. The shape
            of the output of func has to be the same for all elements in arguments.
        arguments (list): List with inputs for func.
        n_cores (int): Number of processes.

    Returns
        evaluations (list): The function evaluations, same length as arguments.

    """
    # extract information
    nan_indices = {
        i for i, arg in enumerate(arguments) if isinstance(arg, float) and np.isnan(arg)
    }
    real_args = [arg for i, arg in enumerate(arguments) if i not in nan_indices]

    # get the batch evaluator if it was provided as string
    if not callable(batch_evaluator):
        batch_evaluator = getattr(
            batch_evaluators, f"{batch_evaluator}_batch_evaluator"
        )

    # evaluate functions
    evaluations = batch_evaluator(
        func=func, arguments=real_args, n_cores=n_cores, error_handling=error_handling
    )

    # combine results
    evaluations = iter(evaluations)
    results = []
    for i in range(len(arguments)):
        if i in nan_indices:
            results.append(np.nan)
        else:
            results.append(next(evaluations))

    return results


def _add_index_to_derivative(derivative, params_index, out_index):
    if len(derivative.shape) == 1 and params_index is not None:
        derivative = pd.Series(derivative, index=params_index)
    if len(derivative.shape) == 2 and (
        params_index is not None or out_index is not None
    ):
        derivative = pd.DataFrame(derivative, columns=params_index, index=out_index)
    return derivative


def _split_into_str_and_int(s):
    """Splits string in str and int parts.

    Args:
        s (str): The string.

    Returns:
        str_part (str): The str part.
        int_part (int): The int part.

    Example:
    >>> s = "forward1"
    >>> _split_into_str_and_int(s)
    ('forward', 1)

    """
    str_part, int_part = re.findall(r"(\w+?)(\d+)", s)[0]
    return str_part, int(int_part)


def _collect_additional_info(return_info, steps, evals, updated_candidates):
    """Combine additional information in dict if return_info is True."""
    info = {}
    if return_info:
        # save function evaluations to accessible data frame
        func_evals = _convert_evaluation_data_to_frame(steps, evals)
        info["func_evals"] = func_evals

        if updated_candidates is not None:
            # combine derivative candidates in accessible data frame
            derivative_candidates = _convert_richardson_candidates_to_frame(
                *updated_candidates
            )
            info["derivative_candidates"] = derivative_candidates

    return info
