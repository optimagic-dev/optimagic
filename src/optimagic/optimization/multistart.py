"""Functions for multi start optimization a la TikTak.

TikTak (`Arnoud, Guvenen, and Kleineberg
<https://www.nber.org/system/files/working_papers/w26340/w26340.pdf>`_)

 is an algorithm for solving global optimization problems. It performs local searches
from a set of carefully-selected points in the parameter space.

First implemented in Python by Alisdair McKay (
`GitHub Repository <https://github.com/amckay/TikTak>`_)

"""

import warnings
from dataclasses import replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc, triang

from optimagic.logging.logger import LogStore
from optimagic.logging.types import StepStatus
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.optimization.multistart_options import InternalMultistartOptions
from optimagic.optimization.optimization_logging import (
    log_scheduled_steps_and_get_ids,
)
from optimagic.typing import AggregationLevel, ErrorHandling
from optimagic.utilities import get_rng


def run_multistart_optimization(
    local_algorithm: Algorithm,
    internal_problem: InternalOptimizationProblem,
    x: NDArray[np.float64],
    sampling_bounds: InternalBounds,
    options: InternalMultistartOptions,
    logger: LogStore | None,
    error_handling: ErrorHandling,
) -> InternalOptimizeResult:
    steps = determine_steps(options.n_samples, stopping_maxopt=options.stopping_maxopt)

    scheduled_steps = log_scheduled_steps_and_get_ids(
        steps=steps,
        logger=logger,
    )

    if options.sample is not None:
        sample = options.sample
    else:
        sample = _draw_exploration_sample(
            x=x,
            lower=sampling_bounds.lower,
            upper=sampling_bounds.upper,
            # -1 because we add start parameters
            n_samples=options.n_samples - 1,
            distribution=options.sampling_distribution,
            method=options.sampling_method,
            seed=options.seed,
        )

        sample = np.vstack([x.reshape(1, -1), sample])

    if logger:
        logger.step_store.update(
            scheduled_steps[0], {"status": StepStatus.RUNNING.value}
        )

    exploration_res = run_explorations(
        internal_problem=internal_problem,
        sample=sample,
        n_cores=options.n_cores,
        step_id=scheduled_steps[0],
    )

    if logger:
        logger.step_store.update(
            scheduled_steps[0], {"status": StepStatus.COMPLETE.value}
        )

    scheduled_steps = scheduled_steps[1:]

    sorted_sample = exploration_res["sorted_sample"]
    sorted_values = exploration_res["sorted_values"]

    stopping_maxopt = options.stopping_maxopt
    if stopping_maxopt > len(sorted_sample):
        n_skipped_steps = stopping_maxopt - len(sorted_sample)
        stopping_maxopt = len(sorted_sample)
        warnings.warn(
            "There are less valid starting points than requested optimizations. "
            "The number of optimizations has been reduced from "
            f"{options.stopping_maxopt} to {len(sorted_sample)}."
        )
        skipped_steps = scheduled_steps[-n_skipped_steps:]
        scheduled_steps = scheduled_steps[:-n_skipped_steps]

        if logger:
            for step in skipped_steps:
                new_status = StepStatus.SKIPPED.value
                logger.step_store.update(step, {"status": new_status})

    batched_sample = get_batched_optimization_sample(
        sorted_sample=sorted_sample,
        stopping_maxopt=stopping_maxopt,
        batch_size=options.batch_size,
    )

    state = {
        "best_x": sorted_sample[0],
        "best_y": sorted_values[0],
        "best_res": None,
        "x_history": [],
        "y_history": [],
        "result_history": [],
        "start_history": [],
    }

    convergence_criteria = {
        "xtol": options.convergence_xtol_rel,
        "max_discoveries": options.convergence_max_discoveries,
    }

    batch_evaluator = options.batch_evaluator

    def single_optimization(x0, step_id):
        """Closure for running a single optimization, given a starting point."""
        problem = internal_problem.with_error_handling(error_handling)
        res = local_algorithm.solve_internal_problem(problem, x0, step_id)
        return res

    opt_counter = 0
    for batch in batched_sample:
        weight = options.weight_func(opt_counter, stopping_maxopt)
        starts = [weight * state["best_x"] + (1 - weight) * x for x in batch]

        arguments = [
            {"x0": x, "step_id": id_}
            for x, id_ in zip(starts, scheduled_steps[: len(batch)], strict=False)
        ]
        scheduled_steps = scheduled_steps[len(batch) :]

        batch_results = batch_evaluator(
            func=single_optimization,
            arguments=arguments,
            unpack_symbol="**",
            n_cores=options.n_cores,
            error_handling=options.error_handling,
        )

        state, is_converged = update_convergence_state(
            current_state=state,
            starts=starts,
            results=batch_results,
            convergence_criteria=convergence_criteria,
            solver_type=local_algorithm.algo_info.solver_type,
        )
        opt_counter += len(batch)
        if is_converged:
            if logger:
                for step in scheduled_steps:
                    new_status = StepStatus.SKIPPED.value
                    logger.step_store.update(step, {"status": new_status})
            break

    multistart_info = {
        "start_parameters": state["start_history"],
        "local_optima": state["result_history"],
        "exploration_sample": sorted_sample,
        "exploration_results": exploration_res["sorted_values"],
    }

    raw_res = state["best_res"]
    res = replace(raw_res, multistart_info=multistart_info)

    return res


def determine_steps(n_samples, stopping_maxopt):
    """Determine the number and type of steps for the multistart optimization.

    This is mainly used to write them to the log. The number of steps is also
    used if logging is False.

    Args:
        n_samples (int): Number of exploration points for the multistart optimization.
        stopping_maxopt (int): Number of local optimizations.


    Returns:
        list: List of dictionaries with information on each step.

    """
    exploration_step = {
        "type": "exploration",
        "status": "running",
        "name": "exploration",
        "n_iterations": n_samples,
    }

    steps = [exploration_step]
    for i in range(stopping_maxopt):
        optimization_step = {
            "type": "optimization",
            "status": "scheduled",
            "name": f"optimization_{i}",
        }
        steps.append(optimization_step)
    return steps


def _draw_exploration_sample(
    x: NDArray[np.float64],
    lower: NDArray[np.float64] | None,
    upper: NDArray[np.float64] | None,
    n_samples: int,
    distribution: Literal["uniform", "triangular"],
    method: Literal["sobol", "random", "halton", "latin_hypercube"],
    seed: int | np.random.Generator | None,
) -> NDArray[np.float64]:
    """Get a sample of parameter values for the first stage of the tiktak algorithm.

    The sample is created randomly or using a low discrepancy sequence. Different
    distributions are available.

    Args:
        x: Internal parameter vector of shape (n_params,).
        lower: Vector of internal lower bounds of shape (n_params,).
        upper: Vector of internal upper bounds of shape (n_params,).
        n_samples: Number of sample points.
        distribution: The distribution from which the exploration sample is
            drawn. Allowed are "uniform" and "triangular". Defaults to "uniform".
        method: The method used to draw the exploration sample. Allowed are
            "sobol", "random", "halton", and "latin_hypercube". Defaults to "sobol".
        seed: Random number seed or generator.

    Returns:
        Array of shape (n_samples, n_params). Each row represents a vector of parameter
            values.

    """
    if lower is None or upper is None:
        raise ValueError("lower and upper bounds must be provided for multistart.")

    for name, bound in zip(["lower", "upper"], [lower, upper], strict=False):
        if not np.isfinite(bound).all():
            raise ValueError(
                f"multistart optimization requires finite {name}_bounds or "
                f"soft_{name}_bounds for all parameters."
            )

    if method == "sobol":
        # Draw `n` points from the open interval (lower, upper)^d.
        # Note that scipy uses the half-open interval [lower, upper)^d internally.
        # We apply a burn-in phase of 1, i.e. we skip the first point in the sequence
        # and thus exclude the lower bound.
        sampler = qmc.Sobol(d=len(lower), scramble=False, seed=seed)
        _ = sampler.fast_forward(1)
        sample_unscaled = sampler.random(n=n_samples)

    elif method == "halton":
        sampler = qmc.Halton(d=len(lower), scramble=False, seed=seed)
        sample_unscaled = sampler.random(n=n_samples)

    elif method == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=len(lower), strength=1, seed=seed)
        sample_unscaled = sampler.random(n=n_samples)

    elif method == "random":
        rng = get_rng(seed)
        sample_unscaled = rng.uniform(size=(n_samples, len(lower)))

    if distribution == "uniform":
        sample_scaled = qmc.scale(sample_unscaled, lower, upper)
    elif distribution == "triangular":
        sample_scaled = triang.ppf(
            sample_unscaled,
            c=(x - lower) / (upper - lower),
            loc=lower,
            scale=upper - lower,
        )

    return sample_scaled


def run_explorations(
    internal_problem: InternalOptimizationProblem,
    sample: NDArray[np.float64],
    n_cores: int,
    step_id: int,
) -> dict[str, NDArray[np.float64]]:
    """Do the function evaluations for the exploration phase.

    Args:
        internal_problem: The internal optimization problem.
        sample: 2d numpy array where each row is a sampled internal
            parameter vector.
        batch_evaluator: See :ref:`batch_evaluators`.
        n_cores: Number of cores.
        step_id: The identifier of the exploration step.

    Returns:
        dict: A dictionary with the the following entries:
            "sorted_values": 1d numpy array with sorted function values. Invalid
                function values are excluded.
            "sorted_sample": 2d numpy array with corresponding internal parameter
                vectors.

    """
    internal_problem = internal_problem.with_step_id(step_id)
    x_list: list[NDArray[np.float64]] = list(sample)

    raw_values = np.asarray(
        internal_problem.exploration_fun(x_list, n_cores=n_cores), dtype=np.float64
    )

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


def get_batched_optimization_sample(sorted_sample, stopping_maxopt, batch_size):
    """Create a batched sample of internal parameters for the optimization phase.

    Note that in the end the optimizations will not be started from those parameter
    vectors but from a convex combination of that parameter vector and the
    best parameter vector at the time when the optimization is started.

    Args:
        sorted_sample (np.ndarray): 2d numpy array with containing sorted internal
            parameter vectors.
        stopping_maxopt (int): Number of optimizations to run. If sample is shorter
            than that, optimizations are run on all entries of the sample.
        batch_size (int): Batch size.

    Returns:
        list: Nested list of parameter vectors from which an optimization is run.
            The inner lists have length ``batch_size`` or shorter.

    """
    n_batches = int(np.ceil(stopping_maxopt / batch_size))

    start = 0
    batched = []
    for _ in range(n_batches):
        stop = min(start + batch_size, len(sorted_sample), stopping_maxopt)
        batched.append(list(sorted_sample[start:stop]))
        start = stop
    return batched


def update_convergence_state(
    current_state, starts, results, convergence_criteria, solver_type
):
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
        solver_type: The aggregation level of the local optimizer. Needed to
            interpret the output of the internal criterion function.


    Returns:
        dict: The updated state, same entries as current_state.
        bool: A bool that indicates if the optimizer has converged.

    """

    # ==================================================================================
    # unpack some variables
    # ==================================================================================
    xtol = convergence_criteria["xtol"]
    max_discoveries = convergence_criteria["max_discoveries"]

    best_x = current_state["best_x"]
    best_y = current_state["best_y"]
    best_res = current_state["best_res"]

    # ==================================================================================
    # filter out optimizations that raised errors
    # ==================================================================================
    # get indices of local optimizations that did not fail
    valid_indices = [i for i, res in enumerate(results) if not isinstance(res, str)]

    # If all local optimizations failed, return early so we don't have to worry about
    # index errors later.
    if not valid_indices:
        return current_state, False
    # ==================================================================================
    # reduce eveything to valid optimizations
    # ==================================================================================
    valid_results = [results[i] for i in valid_indices]
    valid_starts = [starts[i] for i in valid_indices]
    valid_new_x = [res.x for res in valid_results]
    valid_new_y = []

    # make the criterion output scalar if a least squares optimizer returns an
    # array as solution_criterion.
    for res in valid_results:
        if np.isscalar(res.fun):
            fun = float(res.fun)
        elif solver_type == AggregationLevel.LIKELIHOOD:
            fun = float(np.sum(res.fun))
        elif solver_type == AggregationLevel.LEAST_SQUARES:
            fun = np.dot(res.fun, res.fun)

        valid_new_y.append(fun)

    # ==================================================================================
    # accept new best point if we find a new lowest function value
    # ==================================================================================
    best_index = np.argmin(valid_new_y)
    if valid_new_y[best_index] <= best_y:
        best_x = valid_new_x[best_index]
        best_y = valid_new_y[best_index]
        best_res = valid_results[best_index]
    # handle the case that the global optimum was found in the exploration sample and
    # due to floating point imprecisions the result of the optimization that started at
    # the global optimum is slightly worse
    elif best_res is None:
        best_res = valid_results[best_index]

    # ==================================================================================
    # update history and state
    # ==================================================================================
    new_x_history = current_state["x_history"] + valid_new_x
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
        "y_history": current_state["y_history"] + valid_new_y,
        "result_history": current_state["result_history"] + valid_results,
        "start_history": current_state["start_history"] + valid_starts,
    }

    return new_state, is_converged
