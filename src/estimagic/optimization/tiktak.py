"""Functions for multi start optimization a la TikTak.

TikTak (`Arnoud, Guvenen, and Kleineberg
<https://www.nber.org/system/files/working_papers/w26340/w26340.pdf>`_)
is an algorithm for solving global optimization problems. It performs local
searches from a set of carefully-selected points in the parameter space.

First implemented in Python by Alisdair McKay
(`GitHub Repository <https://github.com/amckay/TikTak>`_)

To-Do:

- Make sampling compatible with constraints
    - Write a get_approximate_internal_bounds function
    - Sample on internal parameters space
    - Do not do any bounds checking for fixed parameters

- Improve Error handling
    - Go over all error messages and provide information on the exact problem
      (e.g. section of params where bounds are missing, ...)

"""
import warnings
from functools import partial

import chaospy
import numpy as np
import pandas as pd
from chaospy.distributions import Triangle
from chaospy.distributions import Uniform
from estimagic import batch_evaluators as be
from estimagic.parameters.parameter_conversion import get_internal_bounds
from estimagic.parameters.parameter_conversion import get_reparametrize_functions


def get_exploration_sample(
    params,
    n_samples,
    sampling_distribution,
    sampling_method,
    seed,
    constraints,
):
    """Get a sample of parameter values for the first stage of the tiktak algorithm.

    The sample is created randomly or using low a low discrepancy sequence. Different
    distributions are available.

    Args:
        params (pandas.DataFrame): see :ref:`params`.
        n_samples (int, pandas.DataFrame or numpy.ndarray): Number of sampled points on
            which to do one function evaluation. Default is 10 * n_params.
            Alternatively, a DataFrame or numpy array with an existing sample.
        sampling_distribution (str): One of "uniform", "triangle". Default is
            "uniform"  as in the original tiktak algorithm.
        sampling_method (str): One of "random", "sobol", "halton",
            "hammersley", "korobov", "latin_hypercube" and "chebyshev" or a numpy array
            or DataFrame with custom points. Default is sobol for problems with up to 30
            parameters and random for problems with more than 30 parameters.
        seed (int): Random seed.
        constraints (list): See :ref:`constraints`.

    Returns:
        np.ndarray: Numpy array of shape n_samples, len(params). Each row is a vector
            of parameter values.

    """
    if constraints is None:
        constraints = []

    if isinstance(n_samples, (np.ndarray, pd.DataFrame)):
        sample = _process_sample(n_samples, params, constraints)
    elif isinstance(n_samples, (int, float)):
        sample = _create_sample(
            params=params,
            n_samples=n_samples,
            sampling_distribution=sampling_distribution,
            sampling_method=sampling_method,
            seed=seed,
            constraints=constraints,
        )
    else:
        raise TypeError(f"Invalid type for n_samples: {type(n_samples)}")
    return sample


def _process_sample(raw_sample, params, constraints):
    if isinstance(raw_sample, pd.DataFrame):
        if not raw_sample.columns.equals(params.index):
            raise ValueError(
                "If you provide a custom sample as DataFrame the columns of that "
                "DataFrame and the index of params must be equal."
            )
        sample = raw_sample[params.index].to_numpy()
    elif isinstance(raw_sample, np.ndarray):
        _, n_params = raw_sample.shape
        if n_params != len(params):
            raise ValueError(
                "If you provide a custom sample as a numpy array it must have as many "
                "columns as parameters."
            )
        sample = raw_sample

    to_internal, _ = get_reparametrize_functions(params, constraints)

    sample = np.array([to_internal(x) for x in sample])

    return sample


def _create_sample(
    params,
    n_samples,
    sampling_distribution,
    sampling_method,
    seed,
    constraints,
):
    if _has_transforming_constraints(constraints):
        raise NotImplementedError(
            "Multistart optimization is not yet compatible with transforming "
            "Constraints that require a transformation of parameters such as "
            "linear, probability, covariance and sdcorr constranits."
        )

    lower, upper = _get_internal_sampling_bounds(params, constraints)

    sample = _do_actual_sampling(
        midpoint=params["value"].to_numpy(),
        lower=lower,
        upper=upper,
        size=n_samples,
        distribution=sampling_distribution,
        rule=sampling_method,
        seed=seed,
    )

    return sample


def _do_actual_sampling(midpoint, lower, upper, size, distribution, rule, seed):

    valid_rules = [
        "random",
        "sobol",
        "halton",
        "hammersley",
        "korobov",
        "latin_hypercube",
    ]

    if rule not in valid_rules:
        raise ValueError(f"Invalid rule: {rule}. Must be one of\n\n{valid_rules}\n\n")

    if distribution == "uniform":
        dist_list = [Uniform(lb, ub) for lb, ub in zip(lower, upper)]
    elif distribution == "triangle":
        dist_list = [Triangle(lb, mp, ub) for lb, mp, ub in zip(lower, midpoint, upper)]
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    joint_distribution = chaospy.J(*dist_list)

    np.random.seed(seed)

    sample = joint_distribution.sample(
        size=size,
        rule=rule,
    ).T
    return sample


def _get_internal_sampling_bounds(params, constraints):
    params = params.copy(deep=True)
    params["lower_bound"] = _extract_external_sampling_bound(params, "lower")
    params["upper_bound"] = _extract_external_sampling_bound(params, "upper")

    problematic = params.query("lower_bound >= upper_bound")

    if len(problematic):
        raise ValueError(
            "Lower bound must be smaller than upper bound for all parameters. "
            f"This is violated for:\n\n{problematic.to_string()}\n\n"
        )

    lower, upper = get_internal_bounds(params=params, constraints=constraints)

    for b in lower, upper:
        if not np.isfinite(b).all():
            raise ValueError(
                "Sampling bounds of all free parameters must be finite to create a "
                "parameter sample for multistart optimization."
            )

    return lower, upper


def _extract_external_sampling_bound(params, bounds_type):
    soft_name = f"soft_{bounds_type}_bound"
    hard_name = f"{bounds_type}_bound"
    if soft_name in params:
        bounds = params[soft_name]
    elif hard_name in params:
        bounds = params[hard_name]
    else:
        raise ValueError(
            f"{soft_name} or {hard_name} must be in params to sample start values."
        )

    return bounds


def _has_transforming_constraints(constraints):
    constraints = [] if constraints is None else constraints
    transforming_types = {
        "linear",
        "probability",
        "covariance",
        "sdcorr",
        "increasing",
        "decreasing",
        "sum",
    }
    present_types = {constr["type"] for constr in constraints}
    return bool(transforming_types.intersection(present_types))


def run_explorations(func, sample, batch_evaluator, n_cores):
    """Do the function evaluations for the exploration phase.

    Args:
        func (callable): An already partialled version of
            ``internal_criterion_and_derivative_template`` where the following arguments
            are still free: ``x``, ``task``, ``algorithm_info``, ``error_handling``,
            ``error_penalty``, ``fixed_log_data``.
        sample (numpy.ndarray): 2d numpy array where each row is a sampled internal
            parameter vector.
        batch_evaluator (str or callable): See :ref:`batch_evaluators`.
        n_cores (int): Number of cores.

    Returns:
        dict: A dictionary with the the following entries:
            "sorted_values": 1d numpy array with sorted function values. Invalid
                function values are excluded.
            "sorted_sample": 2d numpy array with corresponding internal parameter
                vectors.
            "contributions": None or 2d numpy array with the contributions entries of
                the function evaluations.
            "root_contributions": None or 2d numpy array with the root_contributions
                entries of the function evaluations.

    """
    algo_info = {
        "primary_criterion_entry": "value",
        "parallelizes": True,
        "needs_scaling": False,
        "name": "tiktak_explorer",
    }

    _func = partial(
        func,
        task="criterion",
        algorithm_info=algo_info,
        error_handling="continue",
        error_penalty={"constant": np.nan, "slope": np.nan},
    )

    arguments = []
    for i, x in enumerate(sample):
        arguments.append(
            {"x": x, "fixed_log_data": {"stage": "exploration", "substage": i}}
        )

    if isinstance(batch_evaluator, str):
        batch_evaluator = getattr(be, f"{batch_evaluator}_batch_evaluator")

    raw_values = batch_evaluator(
        _func, arguments=arguments, n_cores=n_cores, unpack_symbol="**"
    )

    raw_values = np.array(raw_values)

    is_valid = np.isfinite(raw_values)

    if not is_valid.any():
        raise RuntimeError(
            "All function evaluations of the exploration phase in a multistart "
            "optimization are invalid. Check your code or the sampling bounds."
        )

    valid_values = raw_values[is_valid]
    valid_sample = sample[is_valid]

    # this sorts from low to high values; internal criterion and derivative took care
    # of the sign switch.
    sorting_indices = np.argsort(valid_values)

    out = {
        "sorted_values": valid_values[sorting_indices],
        "sorted_sample": valid_sample[sorting_indices],
    }

    return out


def get_batched_optimization_sample(sorted_sample, n_optimizations, batch_size):
    """Create a batched sample of internal parameters for the optimization phase.

    Note that in the end the optimizations will not be started from those parameter
    vectors but from a convex combination of that parameter vector and the
    best parameter vector at the time when the optimization is started.

    Args:
        sorted_sample (np.ndarray): 2d numpy array with containing sorted internal
            parameter vectors.
        n_optimizations (int): Number of optimizations to run. If sample is shorter
            than that, optimizations are run on all entries of the sample.
        batch_size (int): Batch size.

    Returns:
        list: Nested list of parameter vectors from which an optimization is run.
            The inner lists have length ``batch_size`` or shorter.

    """
    if n_optimizations > len(sorted_sample):
        n_optimizations = len(sorted_sample)
        warnings.warn(
            "There are less valid starting points than requested optimizations. "
            "The number of optimizations has been reduced from {n_optimizations} "
            "to {len(sorted_sample)}."
        )

    n_batches = int(np.ceil(n_optimizations / batch_size))

    start = 0
    batched = []
    for _ in range(n_batches):
        stop = min(start + batch_size, len(sorted_sample), n_optimizations)
        batched.append(list(sorted_sample[start:stop]))
        start = stop
    return batched


def run_local_optimizations(
    func,
    sample,
    n_cores,
    batch_evaluator,
    optimize_options,
    mixing_weight_method,
    mixing_weight_bounds,
    convergence_relative_params_tolerance,
):
    """Run the actual local optimizations until convergence.

    Args:
        func (callable): An already partialled version of
            ``internal_criterion_and_derivative_template`` where the following arguments
            are still free: ``x``, ``task``, ``algorithm_info``, ``error_handling``,
            ``error_penalty``, ``fixed_log_data``.
        sample (list): Nested list of parameter vectors from which an optimization is
            run. The inner lists have length ``batch_size`` or shorter.
        n_cores (int): Number of cores.
        batch_evaluator (str or callable): See :ref:`batch_evaluators`.
        optimize_options (dict): Keyword arguments for the optimizations that are fixed
            across all optimizations.
        mixing_weight_method (str or callable): Specifies how much weight is put on the
            currently best point when calculating a new starting point for a local
            optimization out of the currently best point and the next random starting
            point. Either "tiktak" or a callable that takes the arguments ``iteration``,
            ``n_iterations``, ``min_weight``, ``max_weight``. Default "tiktak".
        mixing_weight_bounds (tuple): A tuple consisting of a lower and upper bound on
            mixing weights. Default (0.1, 0.995).
        convergence.relative_params_tolerance (float): If the maximum relative
            difference between the results of two consecutive local optimizations is
            smaller than the multistart optimization converges. Default 0.01. Note that
            this is independent of a convergence criterion with the same name for each
            local optimization.

    Returns:
        dict: A Dictionary containing the best parameters and criterion values,
            convergence information and the history of optimization solutions.

    """
    # process batch evaluator
    # process mixing weight function and options
    # partial optimize options into optimize
    # implement loop with convergence check and parallel optimizations on the inside.
    # process results to something compatible with output of optimize but with
    # additional history entries.


def update_convergence_state(current_state, starts, results, convergence_criteria):
    """Update the state of all quantities related to convergence.

    Args:
        current_state (dict): Dictionary with the entries:
            - "best_x": The currently best parameter vector
            - "best_y": The currently best function value
            - "best_res": The currently best optimization result
            - "x_history": The history of locally optimal parameters
            - "y_history": The history of locally optimal function values.
            - "result_history": The history of local optimization results
            - "start_history": The history of start parameters
        starts (list): List of starting points for local optimizations.
        results (list): List of results from local optimizations.
        convergence_criteria (dict): Dict with the entries "xtol" and "max_discoveries"


    Returns:
        dict: The updated state, same entries as current_state.
        bool: A bool that indicates if the optimizer has converged.

    """
    xtol = convergence_criteria["xtol"]
    max_discoveries = convergence_criteria["max_discoveries"]

    best_x = current_state["best_x"]
    best_y = current_state["best_y"]
    best_res = current_state["best_res"]

    new_x = [res["solution_x"] for res in results]
    new_y = [res["solution_criterion"] for res in results]

    best_index = np.argmin(new_y)
    if new_y[best_index] < best_y:
        best_x = new_x[best_index]
        best_y = new_y[best_index]
        best_res = results[best_index]

    new_x_history = current_state["x_history"] + new_x
    all_x = np.array(new_x_history)
    relative_diffs = (all_x - best_x) / np.clip(best_x, 0.1, np.inf)
    distances = np.linalg.norm(relative_diffs, axis=1)
    n_close = (distances <= xtol).sum()

    is_converged = n_close >= max_discoveries

    new_state = {
        "best_x": best_x,
        "best_y": best_y,
        "best_res": best_res,
        "x_history": new_x_history,
        "y_history": current_state["y_history"] + new_y,
        "result_history": current_state["result_history"] + results,
        "start_history": current_state["start_history"] + starts,
    }

    return new_state, is_converged


def _tiktak_weights(iteration, n_iterations, min_weight, max_weight):
    return np.clip(np.sqrt(iteration / n_iterations), min_weight, max_weight)


def _linear_weights(iteration, n_iterations, min_weight, max_weight):
    unscaled = iteration / n_iterations
    span = max_weight - min_weight
    return min_weight + unscaled * span


WEIGHT_FUNCTIONS = {
    "tiktak": _tiktak_weights,
    "linear": _linear_weights,
}
