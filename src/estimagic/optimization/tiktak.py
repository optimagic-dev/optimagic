"""Functions for multi start optimization a la TikTak.

TikTak (`Arnoud, Guvenen, and Kleineberg
<https://www.nber.org/system/files/working_papers/w26340/w26340.pdf>`_)
is an algorithm for solving global optimization problems. It performs local
searches from a set of carefully-selected points in the parameter space.

First implemented in Python by Alisdair McKay
(`GitHub Repository <https://github.com/amckay/TikTak>`_)
"""
import warnings
from functools import partial

import numpy as np
from estimagic.batch_evaluators import process_batch_evaluator
from estimagic.decorators import AlgoInfo
from estimagic.optimization.optimization_logging import log_scheduled_steps_and_get_ids
from estimagic.optimization.optimization_logging import update_step_status
from estimagic.parameters.conversion import aggregate_func_output_to_value
from scipy.stats import qmc
from scipy.stats import triang


def run_multistart_optimization(
    local_algorithm,
    primary_key,
    problem_functions,
    x,
    lower_sampling_bounds,
    upper_sampling_bounds,
    options,
    logging,
    db_kwargs,
    error_handling,
):
    steps = determine_steps(options["n_samples"], options["n_optimizations"])

    scheduled_steps = log_scheduled_steps_and_get_ids(
        steps=steps,
        logging=logging,
        db_kwargs=db_kwargs,
    )

    if options["sample"] is not None:
        sample = options["sample"]
    else:
        sample = draw_exploration_sample(
            x=x,
            lower=lower_sampling_bounds,
            upper=upper_sampling_bounds,
            # -1 because we add start parameters
            n_samples=options["n_samples"] - 1,
            sampling_distribution=options["sampling_distribution"],
            sampling_method=options["sampling_method"],
            seed=options["seed"],
        )

        sample = np.vstack([x.reshape(1, -1), sample])

    if logging:
        update_step_status(
            step=scheduled_steps[0],
            new_status="running",
            db_kwargs=db_kwargs,
        )

    if "criterion" in problem_functions:
        criterion = problem_functions["criterion"]
    else:
        criterion = partial(list(problem_functions.values())[0], task="criterion")

    exploration_res = run_explorations(
        criterion,
        primary_key=primary_key,
        sample=sample,
        batch_evaluator=options["batch_evaluator"],
        n_cores=options["n_cores"],
        step_id=scheduled_steps[0],
        error_handling=options["exploration_error_handling"],
    )

    if logging:
        update_step_status(
            step=scheduled_steps[0],
            new_status="complete",
            db_kwargs=db_kwargs,
        )

    scheduled_steps = scheduled_steps[1:]

    sorted_sample = exploration_res["sorted_sample"]
    sorted_values = exploration_res["sorted_values"]

    n_optimizations = options["n_optimizations"]
    if n_optimizations > len(sorted_sample):
        n_skipped_steps = n_optimizations - len(sorted_sample)
        n_optimizations = len(sorted_sample)
        warnings.warn(
            "There are less valid starting points than requested optimizations. "
            "The number of optimizations has been reduced from "
            f"{options['n_optimizations']} to {len(sorted_sample)}."
        )
        skipped_steps = scheduled_steps[-n_skipped_steps:]
        scheduled_steps = scheduled_steps[:-n_skipped_steps]

        if logging:
            for step in skipped_steps:
                update_step_status(
                    step=step,
                    new_status="skipped",
                    db_kwargs=db_kwargs,
                )

    batched_sample = get_batched_optimization_sample(
        sorted_sample=sorted_sample,
        n_optimizations=n_optimizations,
        batch_size=options["batch_size"],
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
        "xtol": options["convergence_relative_params_tolerance"],
        "max_discoveries": options["convergence_max_discoveries"],
    }

    problem_functions = {
        name: partial(func, error_handling=error_handling)
        for name, func in problem_functions.items()
    }

    batch_evaluator = options["batch_evaluator"]

    weight_func = partial(
        options["mixing_weight_method"],
        min_weight=options["mixing_weight_bounds"][0],
        max_weight=options["mixing_weight_bounds"][1],
    )

    opt_counter = 0
    for batch in batched_sample:

        weight = weight_func(opt_counter, n_optimizations)
        starts = [weight * state["best_x"] + (1 - weight) * x for x in batch]

        arguments = [
            {**problem_functions, "x": x, "step_id": step}
            for x, step in zip(starts, scheduled_steps)
        ]

        batch_results = batch_evaluator(
            func=local_algorithm,
            arguments=arguments,
            unpack_symbol="**",
            n_cores=options["n_cores"],
            error_handling=options["optimization_error_handling"],
        )

        state, is_converged = update_convergence_state(
            current_state=state,
            starts=starts,
            results=batch_results,
            convergence_criteria=convergence_criteria,
        )
        opt_counter += len(batch)
        scheduled_steps = scheduled_steps[len(batch) :]
        if is_converged:
            if logging:
                for step in scheduled_steps:
                    update_step_status(
                        step=step,
                        new_status="skipped",
                        db_kwargs=db_kwargs,
                    )
            break

    raw_res = state["best_res"]
    raw_res["multistart_info"] = {
        "start_parameters": state["start_history"],
        "local_optima": state["result_history"],
        "exploration_sample": sorted_sample,
        "exploration_results": exploration_res["sorted_values"],
    }

    return raw_res


def determine_steps(n_samples, n_optimizations):
    """Determine the number and type of steps for the multistart optimization.

    This is mainly used to write them to the log. The number of steps is also
    used if logging is False.

    Args:
        n_samples (int): Number of exploration points for the multistart optimization.
        n_optimizations (int): Number of local optimizations.


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
    for i in range(n_optimizations):
        optimization_step = {
            "type": "optimization",
            "status": "scheduled",
            "name": f"optimization_{i}",
        }
        steps.append(optimization_step)
    return steps


def draw_exploration_sample(
    x,
    lower,
    upper,
    n_samples,
    sampling_distribution,
    sampling_method,
    seed,
):
    """Get a sample of parameter values for the first stage of the tiktak algorithm.

    The sample is created randomly or using a low discrepancy sequence. Different
    distributions are available.

    Args:
        x (np.ndarray): Internal parameter vector of shape (n_params,).
        lower (np.ndarray): Vector of internal lower bounds of shape (n_params,).
        upper (np.ndarray): Vector of internal upper bounds of shape (n_params,).
        n_samples (int): Number of sample points on which one function evaluation
            shall be performed. Default is 10 * n_params.
        sampling_distribution (str): One of "uniform", "triangular". Default is
            "uniform", as in the original tiktak algorithm.
        sampling_method (str): One of "sobol", "halton", "latin_hypercube" or
            "random". Default is sobol for problems with up to 200 parameters
            and random for problems with more than 200 parameters.
        seed (int): Random seed.

    Returns:
        np.ndarray: Numpy array of shape (n_samples, n_params).
            Each row represents a vector of parameter values.
    """
    valid_rules = ["sobol", "halton", "latin_hypercube", "random"]
    valid_distributions = ["uniform", "triangular"]

    if sampling_method not in valid_rules:
        raise ValueError(
            f"Invalid rule: {sampling_method}. Must be one of\n\n{valid_rules}\n\n"
        )

    if sampling_distribution not in valid_distributions:
        raise ValueError(f"Unsupported distribution: {sampling_distribution}")

    for name, bound in zip(["lower", "upper"], [lower, upper]):
        if not np.isfinite(bound).all():
            raise ValueError(
                f"multistart optimization requires finite {name}_bounds or "
                f"soft_{name}_bounds for all parameters."
            )

    if sampling_method == "sobol":
        # Draw `n` points from the open interval (lower, upper)^d.
        # Note that scipy uses the half-open interval [lower, upper)^d internally.
        # We apply a burn-in phase of 1, i.e. we skip the first point in the sequence
        # and thus exclude the lower bound.
        sampler = qmc.Sobol(d=len(lower), scramble=False, seed=seed)
        _ = sampler.fast_forward(1)
        sample_unscaled = sampler.random(n=n_samples)

    elif sampling_method == "halton":
        sampler = qmc.Halton(d=len(lower), scramble=False, seed=seed)
        sample_unscaled = sampler.random(n=n_samples)

    elif sampling_method == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=len(lower), strength=1, seed=seed)
        sample_unscaled = sampler.random(n=n_samples)

    elif sampling_method == "random":
        np.random.seed(seed)
        sample_unscaled = np.random.sample(size=(n_samples, len(lower)))

    if sampling_distribution == "uniform":
        sample_scaled = qmc.scale(sample_unscaled, lower, upper)
    elif sampling_distribution == "triangular":
        sample_scaled = triang.ppf(
            sample_unscaled,
            c=(x - lower) / (upper - lower),
            loc=lower,
            scale=upper - lower,
        )

    return sample_scaled


def run_explorations(
    func, primary_key, sample, batch_evaluator, n_cores, step_id, error_handling
):
    """Do the function evaluations for the exploration phase.

    Args:
        func (callable): An already partialled version of
            ``internal_criterion_and_derivative_template`` where the following arguments
            are still free: ``x``, ``task``, ``error_handling``, ``fixed_log_data``.
        primary_key: The primary criterion entry of the local optimizer. Needed to
            interpret the output of the internal criterion function.
        sample (numpy.ndarray): 2d numpy array where each row is a sampled internal
            parameter vector.
        batch_evaluator (str or callable): See :ref:`batch_evaluators`.
        n_cores (int): Number of cores.
        step_id (int): The identifier of the exploration step.
        error_handling (str): One of "raise" or "continue".

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
    algo_info = AlgoInfo(
        primary_criterion_entry=primary_key,
        parallelizes=True,
        needs_scaling=False,
        name="tiktak_explorer",
        is_available=True,
        arguments=[],
    )

    _func = partial(
        func,
        task="criterion",
        algo_info=algo_info,
        error_handling=error_handling,
    )

    arguments = []
    for x in sample:
        arguments.append({"x": x, "fixed_log_data": {"step": int(step_id)}})

    batch_evaluator = process_batch_evaluator(batch_evaluator)

    criterion_outputs = batch_evaluator(
        _func,
        arguments=arguments,
        n_cores=n_cores,
        unpack_symbol="**",
        # If desired, errors are caught inside criterion function.
        error_handling="raise",
    )

    values = [aggregate_func_output_to_value(c, primary_key) for c in criterion_outputs]

    raw_values = np.array(values)

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
    n_batches = int(np.ceil(n_optimizations / batch_size))

    start = 0
    batched = []
    for _ in range(n_batches):
        stop = min(start + batch_size, len(sorted_sample), n_optimizations)
        batched.append(list(sorted_sample[start:stop]))
        start = stop
    return batched


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

    valid_indices = [i for i, res in enumerate(results) if not isinstance(res, str)]

    if not valid_indices:
        return current_state, False

    valid_results = [results[i] for i in valid_indices]
    valid_starts = [starts[i] for i in valid_indices]

    valid_new_x = [res["solution_x"] for res in valid_results]
    valid_new_y = [res["solution_criterion"] for res in valid_results]

    best_index = np.argmin(valid_new_y)
    if valid_new_y[best_index] <= best_y:
        best_x = valid_new_x[best_index]
        best_y = valid_new_y[best_index]
        best_res = valid_results[best_index]

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
