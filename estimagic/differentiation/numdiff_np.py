import functools
from collections import OrderedDict
from itertools import product

import numpy as np
from joblib import delayed
from joblib import Parallel

from estimagic.decorators import de_scalarize
from estimagic.decorators import nan_if_exception
from estimagic.differentiation import finite_differences
from estimagic.differentiation.generate_steps import generate_steps
from estimagic.differentiation.richardson_extrapolation import richardson_extrapolation
from estimagic.optimization.utilities import namedtuple_from_kwargs


def first_derivative(
    func,
    x,
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
    n_cores=1,
    return_richardson_info=False,
):

    """Evaluate first derivative of func at x according to method and step options.

    Internally, the function is converted such that it maps from a 1d array to a 1d
    array. Then the Jacobian of that function is calculated. The resulting derivative
    estimate is always a numpy array.

    Detailed description of all options that influence the step size as well as an
    explanation of how steps are adjusted to bounds in case of a conflict,
    see :func:`~estimagic.differentiation.generate_steps.generate_steps`.

    Args:
        func (callable): Function of which the derivative is calculated.
        x (np.ndarray): 1d array at which the derivative is calculated.
        func_kwargs (dict): Additional keyword arguments for func, optional.
        method (str): One of ["central", "forward", "backward"], default "central".
        n_steps (int): Number of steps needed. For central methods, this is
            the number of steps per direction. It is 1 if no Richardson extrapolation
            is used.
        base_steps (np.ndarray, optional): 1d array of the same length as x. base_steps
            * scaling_factor is the absolute value of the first (and possibly only) step
            used in the finite differences approximation of the derivative. If the
            base_steps * scaling_factor conflicts with bounds, the actual steps will
            be adjusted. If base_steps is not provided, it will be determined according
            to a rule of thumb as long as this does not conflict with min_steps.
        scaling_factor (np.ndarray or float): Scaling factor which is applied to
            base_steps. If it is an np.ndarray, it needs to have the same shape as x.
            scaling_factor is useful if you want to increase or decrease the base_step
            relative to the rule-of-thumb or user provided base_step, for example to
            benchmark the effect of the step size. Default 1.
        lower_bounds (np.ndarray): 1d array with lower bounds for each parameter.
        upper_bounds (np.ndarray): 1d array with upper bounds for each parameter.
        step_ratio (float or array): Ratio between two consecutive Richardson
            extrapolation steps in the same direction. default 2.0. Has to be larger
            than one. step ratio is only used if n_steps > 1.
        min_steps (np.ndarray): Minimal possible step sizes that can be chosen to
            accommodate bounds. Needs to have same length as x. By default min_steps is
            equal to base_steps, i.e step size is not decreased beyond what is optimal
            according to the rule of thumb.
        f0 (np.ndarray): 1d numpy array with func(x), optional.
        n_cores (int): Number of processes used to parallelize the function
            evaluations. Default 1.
        return_richardson_info (bool): Should additional information on the Richardson
            extrapolation be returned. Has no effect if n_steps = 1.

    Returns:
        derivative (np.ndarray): The estimated first derivative of func at x.
            The shape of the output depends on the dimension of x and func(x):
            f: R -> R leads to shape (1,), usually called derivative
            f: R^m -> R leads to shape (m, ), usually called Gradient
            f: R -> R^n leads to shape (n, 1), usually called Jacobian
            f: R^m -> R^n leads to shape (n, m), usually called Jacobian

        info (OrderedDict): Dictionary with all derivative estimates and
            error estimates for different parameter specifications using Richardson
            extrapolations. Is only returned if return_richardson_info is True.

    """
    func_kwargs = {} if func_kwargs is None else func_kwargs
    partialed_func = functools.partial(func, **func_kwargs)
    f0 = partialed_func(x) if f0 is None else f0

    x_was_scalar = np.isscalar(x)
    f_was_scalar = np.isscalar(f0)

    x = np.atleast_1d(x).astype(np.float_)
    f0 = np.atleast_1d(f0).astype(np.float_)

    @nan_if_exception
    @de_scalarize(x_was_scalar)
    def internal_func(x):
        return partialed_func(x)

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

    evaluation_points = []
    for step_arr in steps:
        for i, j in product(range(n_steps), range(len(x))):
            if np.isnan(step_arr[i, j]):
                evaluation_points.append(np.nan)
            else:
                point = x.copy()
                point[j] += step_arr[i, j]
                evaluation_points.append(point)

    raw_evals = _nan_skipping_batch_evaluator(internal_func, evaluation_points, n_cores)

    evals = np.array(raw_evals).reshape(2, n_steps, len(x), -1)
    evals = np.transpose(evals, axes=(0, 1, 3, 2))
    evals = namedtuple_from_kwargs(pos=evals[0], neg=evals[1])

    jac_candidates = {}
    for m in ["forward", "backward", "central"]:
        jac_candidates[m] = finite_differences.jacobian(evals, steps, f0, method)

    orders = {
        "central": ["central", "forward", "backward"],
        "forward": ["forward", "backward"],
        "backward": ["backward", "forward"],
    }

    if n_steps == 1:
        jac = _consolidate_one_step_derivatives(jac_candidates, orders[method])
    else:
        richardson_candidates = _compute_richardson_candidates(
            jac_candidates, steps, n_steps
        )
        jac = _consolidate_extrapolated(richardson_candidates)

    derivative = jac.flatten() if f_was_scalar else jac

    return_info = n_steps > 1 and return_richardson_info
    out = (derivative, richardson_candidates) if return_info else derivative
    return out


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

    """
    # first find minimum over steps for each method
    candidate_derivatives = OrderedDict()
    candidate_errors = OrderedDict()

    for key in candidates.keys():
        _limit = candidates[key]["derivative"]
        _error = candidates[key]["error"]

        derivative, error = _get_best_estimate_single_method(_limit, _error)

        candidate_derivatives[key] = derivative
        candidate_errors[key] = error

    # second find minimum over methods
    consolidated = _get_best_estimate_along_methods(
        candidate_derivatives, candidate_errors
    )
    return consolidated


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


def _get_best_estimate_single_method(derivative, errors):
    """Select best derivative estimates element wise.

    Given a single method, e.g. central differences with 2 num_terms (see above), we get
    multiple Richardson approximations including estimated errors. Here we select the
    approximations which result in the lowest error element wise. However, since it can
    be that one derivative estimate is only an outlier, we adjust for outliers.

    Args:
        derivative (np.ndarray): Derivative estimates from Richardson approximation.
            First axis (axis 0) denotes the potentially multiple estimates. Following
            dimensions represent the dimension of the derivative, i.e. for a classical
            gradient ``derivative`` has 2 dimensions, while for a classical jacobian
            ``derivative`` has 3 dimensions.

        errors (np.ndarray): Error estimates of ``derivative`` estimates. Has the same
            shape as ``derivative``.

    Returns:
        - derivative_minimal (np.ndarray): Best derivate estimates chosen with respect
            to minimizing ``errors``. Note that the best values are selected
            element-wise. Has shape ``(derivative.shape[1], derivative.shape[2])``.

        - error_minimal (np.ndarray): Minimal errors selected element-wise along axis
            0 of ``errors``.

    """
    if derivative.shape[0] == 1:
        derivative_minimal = np.squeeze(derivative, axis=0)
        error_minimal = np.squeeze(errors, axis=0)
    else:
        try:
            minimizer = np.nanargmin(errors, axis=0)

            derivative_minimal = np.take_along_axis(
                derivative, minimizer[np.newaxis, :], axis=0
            )
            derivative_minimal = np.squeeze(derivative_minimal, axis=0)
            error_minimal = np.nanmin(errors, axis=0)
        except ValueError:
            raise ValueError(
                "Richardson extrapolation failing --cannot compute minimum of errors "
                "along axis 0. "
            )

    return derivative_minimal, error_minimal


def _get_best_estimate_along_methods(derivatives, errors):
    """Extract best derivative estimate over different methods.

    Given that for each method, where one method can be for example central differences
    with two num_terms (see above), we have selected a single best derivative estimate,
    we select the best derivative estimates element-wise over different methods, where
    again best is defined as minimizing the approximation error.

    Args:
        derivatives (OrderedDict): Dictionary containing derivative estimates for
            different methods.

        errors (OrderedDict): Dictionary containing error estimates for derivates stored
            in ``derivatives``.

    Returns:
        jac_minimal (np.ndarray): The optimal derivative estimate over different
            methods.

    """
    errors = np.stack(list(errors.values()))
    derivatives = np.stack(list(derivatives.values()))

    if derivatives.shape[0] == 1:
        jac_minimal = np.squeeze(derivatives, axis=0)
    else:
        try:
            minimizer = np.nanargmin(errors, axis=0)

            jac_minimal = np.take_along_axis(
                derivatives, minimizer[np.newaxis, :], axis=0
            )
            jac_minimal = np.squeeze(jac_minimal, axis=0)
        except ValueError:
            raise ValueError(
                "Richardson extrapolation failing --cannot compute minimum of errors "
                "along axis 0. "
            )

    return jac_minimal


def _nan_skipping_batch_evaluator(func, arglist, n_cores):
    """Evaluate func at each entry in arglist, skipping np.nan entries.

    The function is only evaluated at inputs that are not a scalar np.nan.
    The outputs corresponding to skipped inputs as well as for inputs on which func
    returns a scalar np.nan are arrays of the same shape as the result of func, filled
    with np.nan.

    Args:
        func (function): Python function that returns a numpy array. The shape
            of the output of func has to be the same for all elements in arglist.
        arglist (list): List with inputs for func.
        n_cores (int): Number of processes.

    Returns
        evaluations (list): The function evaluations, same length as arglist.

    """
    # extract information
    nan_indices = {
        i for i, arg in enumerate(arglist) if isinstance(arg, float) and np.isnan(arg)
    }
    real_args = [arg for i, arg in enumerate(arglist) if i not in nan_indices]

    # evaluate function
    evaluations = Parallel(n_jobs=n_cores)(delayed(func)(point) for point in real_args)

    # combine results
    evaluations = iter(evaluations)
    results = []
    for i in range(len(arglist)):
        if i in nan_indices:
            results.append(np.nan)
        else:
            results.append(next(evaluations))

    # replace scalar NaNs by arrays filled with NaNs.
    outshape = _get_output_shape(results)
    for i in range(len(results)):
        if isinstance(results[i], float) and np.isnan(results[i]):
            results[i] = np.full(outshape, np.nan)

    return results


def _get_output_shape(evals):
    """Get the output shape of func from evaluations.

    Args:
        evals (list): Contains np.nan and numpy arrays that all have the same shape.

    Returns:
        tuple: The shape of the numpy arrays.

    """
    first_relevant = next(x for x in evals if hasattr(x, "shape"))
    return first_relevant.shape
