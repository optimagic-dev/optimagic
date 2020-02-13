from multiprocessing import Pool

import numpy as np

import estimagic.differentiation.finite_differences as fd
from estimagic.differentiation.generate_steps import generate_steps
from estimagic.optimization.utilities import namedtuple_from_kwargs


def _jacobian(
    func,
    x,
    method,
    n_steps,
    base_step,
    lower_bound,
    upper_bound,
    min_step,
    step_ratio,
    f0,
    n_processes,
):

    """
    func catches exceptions and return a scalar nan in that case.
    """
    assert (
        upper_bound - lower_bound >= 2 * min_step
    ).all(), "min_step is too large to fit into bounds."

    steps = generate_steps(
        x=x,
        method=method,
        n_steps=n_steps,
        target="jacobian",
        base_step=base_step,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        step_ratio=step_ratio,
        min_step=min_step,
    )

    evaluation_points = []
    for step_arr in steps:
        for i in range(n_steps):
            for j in range(len(x)):
                if np.isfinite(step_arr[i, j]):
                    point = x.copy()
                    point[j] += step_arr[i, j]
                    evaluation_points.append(point)
                else:
                    evaluation_points.append(np.nan)

    evaluations = _nan_skipping_batch_evaluator(func, evaluation_points, n_processes)

    evals = np.array(evaluations).reshape(2, n_steps, len(x), -1)
    evals = np.transpose(evals, axes=(0, 1, 3, 2))
    evals = namedtuple_from_kwargs(pos=evals[0], neg=evals[1])

    jac_candidates = {}
    for m in ["forward", "backward", "central"]:
        jac_candidates[m] = getattr(fd, f"jacobian_{m}")(evals, steps, f0)

    if n_steps == 1:
        if method in ["forward", "backward"]:
            jac = _consolidate_one_step_one_sided(jac_candidates)
        else:
            jac = _consolidate_one_step_two_sided(jac_candidates)
    else:
        raise NotImplementedError("Extrapolation is not yet implemented.")

    return jac


def _consolidate_one_step_one_sided(candidates):
    consolidated = _fill_nans_with_other(candidates["forward"], candidates["backward"])
    return consolidated.reshape(consolidated.shape[1:])


def _consolidate_one_step_two_sided(candidates):
    consolidated = candidates["central"]
    for other in ["forward", "backward"]:
        consolidated = _fill_nans_with_other(consolidated, candidates[other])
    return consolidated.reshape(consolidated.shape[1:])


def _consolidate_extrapolated(candidates):
    raise NotImplementedError


def _fill_nans_with_other(arr, other):
    assert arr.shape == other.shape, "arr and other must have same shape."
    return np.where(np.isfinite(arr), arr, other)


def _nan_skipping_batch_evaluator(func, arglist, n_processes):
    """Evaluate func at each entry in arglist, skipping np.nan entries.

    Args:
        func (function): Python function that returns a numpy array. The shape
            of the output of func has to be the same for all elements in arglist.
        arglist (list): List with inputs for func.
        n_processes (int): Number of processes.

    Returns
        evaluations (list): The function evaluations, same length as arglist.

    The functions is only evaluated at inputs that are not a scalar np.nan.
    The outputs for the skipped inputs are arrays of the same shape as
    the result of func, filled with np.nan.

    """
    # extract information
    nan_indices = []
    real_args = []
    for i, arg in enumerate(arglist):
        if isinstance(arg, float) and np.isnan(arg):
            nan_indices.append(i)
        else:
            real_args.append(arg)
    nan_indices = set(nan_indices)

    # evaluate function
    if n_processes == 1:
        evaluations = []
        for point in real_args:
            evaluations.append(func(point))
    else:
        p = Pool(processes=n_processes)
        evaluations = p.map(func, real_args)

    # combine results
    outshape = _get_output_shape(evaluations)
    evaluations = iter(evaluations)
    results = []
    for i in range(len(arglist)):
        if i in nan_indices:
            results.append(np.full(outshape, np.nan))
        else:
            evaluation = next(evaluations)
            if isinstance(evaluation, float) and np.isnan(evaluation):
                results.append(np.full(outshape, np.nan))
            else:
                results.append(evaluation)

    return results


def _get_output_shape(evals):
    first_relevant = next(x for x in evals if hasattr(x, "shape"))
    return first_relevant.shape
