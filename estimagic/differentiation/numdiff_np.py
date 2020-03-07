import functools
from itertools import product

import numpy as np
from joblib import delayed
from joblib import Parallel

from estimagic.decorators import de_scalarize
from estimagic.decorators import nan_if_exception
from estimagic.differentiation import finite_differences
from estimagic.differentiation.generate_steps import generate_steps
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

    Returns:
        derivative (np.ndarray): The estimated first derivative of func at x.
            The shape of the output depends on the dimension of x and func(x):
            f: R -> R leads to shape (1,), usually called derivative
            f: R^m -> R leads to shape (m, ), usually called Gradient
            f: R -> R^n leads to shape (n, 1), usually called Jacobian
            f: R^m -> R^n leads to shape (n, m), usually called Jacobian

    """
    func_kwargs = {} if func_kwargs is None else func_kwargs
    partialed_func = functools.partial(func, **func_kwargs)
    f0 = partialed_func(x) if f0 is None else f0

    x_was_scalar = np.isscalar(x)
    f_was_scalar = np.isscalar(f0)

    x = np.atleast_1d(x)
    f0 = np.atleast_1d(f0)

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
        raise NotImplementedError("Extrapolation is not yet implemented.")

    res = jac.flatten() if f_was_scalar else jac

    return res


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
        consolidated = np.where(
            np.isnan(consolidated), candidates[other], consolidated
        )

    return consolidated.reshape(consolidated.shape[1:])


def _consolidate_extrapolated(candidates):
    """Get the best possible derivative estimate, given an error estimate.

    See https://tinyurl.com/ubn3nv5 for corresponding code in numdifftools and
    https://tinyurl.com/snle7mb for an explanation of how errors of Richardson
    extrapolated derivative estimates can be estimated.

    """
    raise NotImplementedError


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
