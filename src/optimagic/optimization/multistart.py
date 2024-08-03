"""Functions for multi start optimization a la TikTak.

TikTak (`Arnoud, Guvenen, and Kleineberg
<https://www.nber.org/system/files/working_papers/w26340/w26340.pdf>`_)

 is an algorithm for solving global optimization problems. It performs local searches
from a set of carefully-selected points in the parameter space.

First implemented in Python by Alisdair McKay (
`GitHub Repository <https://github.com/amckay/TikTak>`_)

"""

import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Sequence, TypedDict, get_args

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc, triang
from typing_extensions import NotRequired

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.decorators import AlgoInfo
from optimagic.exceptions import InvalidMultistartError
from optimagic.optimization.optimization_logging import (
    log_scheduled_steps_and_get_ids,
    update_step_status,
)
from optimagic.parameters.conversion import aggregate_func_output_to_value
from optimagic.typing import PyTree
from optimagic.utilities import get_rng

# ======================================================================================
# Multistart Options Handling
# ======================================================================================

# Public Options
# ======================================================================================

MultistartSamplingMethod = Literal["sobol", "random", "halton", "latin_hypercube"]


@dataclass
class MultistartOptions:
    """Multistart options in optimization problems.

    Attributes:
        n_samples: The number of points at which the objective function is evaluated
            during the exploration phase. If None, n_samples is set to 100 times the
            number of parameters.
        n_optimizations: The number of local optimizations to run. Defaults to 10% of
            n_samples.
        sampling_distribution: The distribution from which the exploration sample is
            drawn. Allowed are "uniform" and "triangular". Defaults to "uniform".
        sampling_method: The method used to draw the exploration sample. Allowed are
            "sobol", "random", "halton", and "latin_hypercube". Defaults to "random".
        sample: A sequence of PyTrees or None. If None, a sample is drawn from the
            sampling distribution.
        mixing_weight_method: The method used to determine the mixing weight, i,e, how
            start parameters for local optimizations are calculated. Allowed are
            "tiktak" and "linear", or a custom callable. Defaults to "tiktak".
        mixing_weight_bounds: The lower and upper bounds for the mixing weight.
            Defaults to (0.1, 0.995).
        convergence_max_discoveries: The maximum number of discoveries for convergence.
            Determines after how many re-descoveries of the currently best local
            optima the multistart algorithm stops. Defaults to 2.
        convergence_relative_params_tolerance: The relative tolerance in parameters
            for convergence. Determines the maximum relative distance two parameter
            vecctors can have to be considered equal. Defaults to 0.01.
        n_cores: The number of cores to use for parallelization. Defaults to 1.
        batch_evaluator: The evaluator to use for batch evaluation. Allowed are "joblib"
            and "pathos", or a custom callable.
        batch_size: The batch size for batch evaluation. Must be larger than n_cores
            or None.
        seed: The seed for the random number generator.
        exploration_error_handling: The error handling for exploration errors. Allowed
            are "raise" and "continue".
        optimization_error_handling: The error handling for optimization errors. Allowed
            are "raise" and "continue".

    Raises:
        InvalidMultistartError: If the multistart options cannot be processed, e.g.
            because they do not have the correct type.

    """

    n_samples: int | None = None
    n_optimizations: int | None = None
    sampling_distribution: Literal["uniform", "triangular"] = "uniform"
    sampling_method: MultistartSamplingMethod = "random"
    sample: Sequence[PyTree] | None = None
    mixing_weight_method: Literal["tiktak", "linear"] = "tiktak"
    mixing_weight_bounds: tuple[float, float] = (0.1, 0.995)
    convergence_relative_params_tolerance: float = 0.01
    convergence_max_discoveries: int = 2
    n_cores: int = 1
    batch_evaluator: Literal["joblib", "pathos"] | Callable = "joblib"
    batch_size: int | None = None
    seed: int | np.random.Generator | None = None
    exploration_error_handling: Literal["raise", "continue"] = "continue"
    optimization_error_handling: Literal["raise", "continue"] = "continue"

    def __post_init__(self) -> None:
        _validate_attribute_types_and_values(self)


class MultistartOptionsDict(TypedDict):
    n_samples: NotRequired[int | None]
    n_optimizations: NotRequired[int | None]
    sampling_distribution: NotRequired[Literal["uniform", "triangular"]]
    sampling_method: NotRequired[MultistartSamplingMethod]
    sample: NotRequired[Sequence[PyTree] | None]
    mixing_weight_method: NotRequired[Literal["tiktak", "linear"]]
    mixing_weight_bounds: NotRequired[tuple[float, float]]
    convergence_relative_params_tolerance: NotRequired[float]
    convergence_max_discoveries: NotRequired[int]
    n_cores: NotRequired[int]
    batch_evaluator: NotRequired[Literal["joblib", "pathos"] | Callable]
    batch_size: NotRequired[int | None]
    seed: NotRequired[int | np.random.Generator | None]
    exploration_error_handling: NotRequired[Literal["raise", "continue"]]
    optimization_error_handling: NotRequired[Literal["raise", "continue"]]


def pre_process_multistart(
    multistart: bool | MultistartOptions | MultistartOptionsDict | None,
) -> MultistartOptions | None:
    """Convert all valid types of multistart to a optimagic.MultistartOptions.

    This just harmonizes multiple ways of specifying multistart options into a single
    format. It performs runime type checks, but it does not check whether multistart
    options are consistent with other option choices.

    Args:
        multistart: The user provided multistart options.
        n_params: The number of parameters in the optimization problem.

    Returns:
        The multistart options in the optimagic format.

    Raises:
        InvalidMultistartError: If the multistart options cannot be processed, e.g.
            because they do not have the correct type.

    """
    if isinstance(multistart, bool):
        multistart = MultistartOptions() if multistart else None
    elif isinstance(multistart, MultistartOptions) or multistart is None:
        pass
    else:
        try:
            multistart = MultistartOptions(**multistart)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if isinstance(e, InvalidMultistartError):
                raise e
            raise InvalidMultistartError(
                f"Invalid multistart options of type: {type(multistart)}. Multistart "
                "options must be of type optimagic.MultistartOptions, a dictionary "
                "with valid keys, None, or a boolean."
            ) from e

    return multistart


def _validate_attribute_types_and_values(options: MultistartOptions) -> None:
    if options.n_samples is not None and (
        not isinstance(options.n_samples, int) or options.n_samples < 1
    ):
        raise InvalidMultistartError(
            f"Invalid number of samples: {options.n_samples}. Number of samples "
            "must be a positive integer or None."
        )

    if options.n_optimizations is not None and (
        not isinstance(options.n_optimizations, int) or options.n_optimizations < 0
    ):
        raise InvalidMultistartError(
            f"Invalid number of optimizations: {options.n_optimizations}. Number of "
            "optimizations must be a positive integer or None."
        )

    if (
        options.n_samples is not None
        and options.n_optimizations is not None
        and options.n_samples < options.n_optimizations
    ):
        raise InvalidMultistartError(
            f"Invalid number of samples: {options.n_samples}. Number of samples "
            "must be at least as large as the number of optimizations."
        )

    if options.sampling_distribution not in ("uniform", "triangular"):
        raise InvalidMultistartError(
            f"Invalid sampling distribution: {options.sampling_distribution}. Sampling "
            "distribution must be 'uniform' or 'triangular'."
        )

    if options.sampling_method not in get_args(MultistartSamplingMethod):
        raise InvalidMultistartError(
            f"Invalid sampling method: {options.sampling_method}. Sampling method "
            f"must be one of {get_args(MultistartSamplingMethod)}."
        )

    if not isinstance(options.sample, Sequence | None):
        raise InvalidMultistartError(
            f"Invalid sample: {options.sample}. Sample must be a sequence of "
            "parameters."
        )

    if not callable(
        options.mixing_weight_method
    ) and options.mixing_weight_method not in ("tiktak", "linear"):
        raise InvalidMultistartError(
            f"Invalid mixing weight method: {options.mixing_weight_method}. Mixing "
            "weight method must be Callable or one of 'tiktak' or 'linear'."
        )

    if (
        not isinstance(options.mixing_weight_bounds, tuple)
        or len(options.mixing_weight_bounds) != 2
        or not set(type(x) for x in options.mixing_weight_bounds) <= {int, float}
    ):
        raise InvalidMultistartError(
            f"Invalid mixing weight bounds: {options.mixing_weight_bounds}. Mixing "
            "weight bounds must be a tuple of two numbers."
        )

    if not isinstance(options.convergence_relative_params_tolerance, int | float):
        raise InvalidMultistartError(
            "Invalid relative params tolerance:"
            f"{options.convergence_relative_params_tolerance}. Relative params "
            "tolerance must be a number."
        )

    if (
        not isinstance(options.convergence_max_discoveries, int | float)
        or options.convergence_max_discoveries < 1
    ):
        raise InvalidMultistartError(
            f"Invalid max discoveries: {options.convergence_max_discoveries}. Max "
            "discoveries must be a positive integer or infinity."
        )

    if not isinstance(options.n_cores, int) or options.n_cores < 1:
        raise InvalidMultistartError(
            f"Invalid number of cores: {options.n_cores}. Number of cores "
            "must be a positive integer."
        )

    if not callable(options.batch_evaluator) and options.batch_evaluator not in (
        "joblib",
        "pathos",
    ):
        raise InvalidMultistartError(
            f"Invalid batch evaluator: {options.batch_evaluator}. Batch evaluator "
            "must be a Callable or one of 'joblib' or 'pathos'."
        )

    if options.batch_size is not None and (
        not isinstance(options.batch_size, int) or options.batch_size < options.n_cores
    ):
        raise InvalidMultistartError(
            f"Invalid batch size: {options.batch_size}. Batch size "
            "must be a positive integer larger than n_cores, or None."
        )

    if not isinstance(options.seed, int | np.random.Generator | None):
        raise InvalidMultistartError(
            f"Invalid seed: {options.seed}. Seed "
            "must be an integer, a numpy random generator, or None."
        )

    if options.exploration_error_handling not in ("raise", "continue"):
        raise InvalidMultistartError(
            f"Invalid exploration error handling: {options.exploration_error_handling}."
            " Exploration error handling must be 'raise' or 'continue'."
        )

    if options.optimization_error_handling not in ("raise", "continue"):
        raise InvalidMultistartError(
            "Invalid optimization error handling:"
            f"{options.optimization_error_handling}. Optimization error handling must "
            "be 'raise' or 'continue'."
        )


# Internal Options
# ======================================================================================


@dataclass
class InternalMultistartOptions:
    """Multistart options used internally in optimagic.

    Compared to `MultistartOptions`, this data class has stricter types and combines
    some of the attributes. It is generated at runtime using a `MultistartOptions`
    instance and the function `get_internal_multistart_options_from_public`.

    """

    n_samples: int
    # TODO: Sampling distribution and method can potentially be combined
    sampling_distribution: Literal["uniform", "triangular"]
    sampling_method: MultistartSamplingMethod
    sample: NDArray[np.float64] | None
    # TODO: Add more informative type hint for weight_func
    weight_func: Callable
    convergence_relative_params_tolerance: float
    convergence_max_discoveries: int
    n_cores: int
    # TODO: Add more informative type hint for batch_evaluator
    batch_evaluator: Callable
    batch_size: int
    seed: int | np.random.Generator | None
    exploration_error_handling: Literal["raise", "continue"]
    optimization_error_handling: Literal["raise", "continue"]
    n_optimizations: int


def get_internal_multistart_options_from_public(
    options: MultistartOptions,
    params: PyTree,
    params_to_internal: Callable[[PyTree], NDArray[np.float64]],
) -> InternalMultistartOptions:
    """Get internal multistart options from public multistart options.

    Args:
        options: The pre-processed multistart options.
        params: The parameters of the optimization problem.
        params_to_internal: A function that converts parameters to internal parameters.

    Returns:
        InternalMultistartOptions: The updated options with runtime defaults.

    """
    x = params_to_internal(params)

    if options.sample is not None:
        sample = np.array([params_to_internal(x) for x in list(options.sample)])
        n_samples = len(options.sample)
    else:
        sample = None
        n_samples = options.n_samples  # type: ignore

    batch_size = options.n_cores if options.batch_size is None else options.batch_size
    batch_evaluator = process_batch_evaluator(options.batch_evaluator)

    if isinstance(options.mixing_weight_method, str):
        _weight_method = WEIGHT_FUNCTIONS[options.mixing_weight_method]
    else:
        _weight_method = options.mixing_weight_method
    weight_func = partial(
        _weight_method,
        min_weight=options.mixing_weight_bounds[0],
        max_weight=options.mixing_weight_bounds[1],
    )

    if n_samples is None:
        if options.n_optimizations is None:
            n_samples = 100 * len(x)
        else:
            n_samples = 10 * options.n_optimizations

    if options.n_optimizations is None:
        n_optimizations = max(1, int(0.1 * n_samples))
    else:
        n_optimizations = options.n_optimizations

    return InternalMultistartOptions(
        # Attributes taken directly from MultistartOptions
        sampling_method=options.sampling_method,
        sampling_distribution=options.sampling_distribution,
        convergence_relative_params_tolerance=options.convergence_relative_params_tolerance,
        convergence_max_discoveries=options.convergence_max_discoveries,
        n_cores=options.n_cores,
        seed=options.seed,
        exploration_error_handling=options.exploration_error_handling,
        optimization_error_handling=options.optimization_error_handling,
        # Updated attributes
        n_samples=n_samples,
        sample=sample,
        weight_func=weight_func,
        n_optimizations=n_optimizations,
        batch_evaluator=batch_evaluator,
        batch_size=batch_size,
    )


# ======================================================================================
# Multistart Optimization
# ======================================================================================


def run_multistart_optimization(
    local_algorithm,
    primary_key,
    problem_functions,
    x,
    lower_sampling_bounds,
    upper_sampling_bounds,
    options,
    logging,
    database,
    error_handling,
):
    steps = determine_steps(options.n_samples, n_optimizations=options.n_optimizations)

    scheduled_steps = log_scheduled_steps_and_get_ids(
        steps=steps,
        logging=logging,
        database=database,
    )

    if options.sample is not None:
        sample = options.sample
    else:
        sample = draw_exploration_sample(
            x=x,
            lower=lower_sampling_bounds,
            upper=upper_sampling_bounds,
            # -1 because we add start parameters
            n_samples=options.n_samples - 1,
            sampling_distribution=options.sampling_distribution,
            sampling_method=options.sampling_method,
            seed=options.seed,
        )

        sample = np.vstack([x.reshape(1, -1), sample])

    if logging:
        update_step_status(
            step=scheduled_steps[0],
            new_status="running",
            database=database,
        )

    if "criterion" in problem_functions:
        criterion = problem_functions["criterion"]
    else:
        criterion = partial(list(problem_functions.values())[0], task="criterion")

    exploration_res = run_explorations(
        criterion,
        primary_key=primary_key,
        sample=sample,
        batch_evaluator=options.batch_evaluator,
        n_cores=options.n_cores,
        step_id=scheduled_steps[0],
        error_handling=options.exploration_error_handling,
    )

    if logging:
        update_step_status(
            step=scheduled_steps[0],
            new_status="complete",
            database=database,
        )

    scheduled_steps = scheduled_steps[1:]

    sorted_sample = exploration_res["sorted_sample"]
    sorted_values = exploration_res["sorted_values"]

    n_optimizations = options.n_optimizations
    if n_optimizations > len(sorted_sample):
        n_skipped_steps = n_optimizations - len(sorted_sample)
        n_optimizations = len(sorted_sample)
        warnings.warn(
            "There are less valid starting points than requested optimizations. "
            "The number of optimizations has been reduced from "
            f"{options.n_optimizations} to {len(sorted_sample)}."
        )
        skipped_steps = scheduled_steps[-n_skipped_steps:]
        scheduled_steps = scheduled_steps[:-n_skipped_steps]

        if logging:
            for step in skipped_steps:
                update_step_status(
                    step=step,
                    new_status="skipped",
                    database=database,
                )

    batched_sample = get_batched_optimization_sample(
        sorted_sample=sorted_sample,
        n_optimizations=n_optimizations,
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
        "xtol": options.convergence_relative_params_tolerance,
        "max_discoveries": options.convergence_max_discoveries,
    }

    problem_functions = {
        name: partial(func, error_handling=error_handling)
        for name, func in problem_functions.items()
    }

    batch_evaluator = options.batch_evaluator

    opt_counter = 0
    for batch in batched_sample:
        weight = options.weight_func(opt_counter, n_optimizations)
        starts = [weight * state["best_x"] + (1 - weight) * x for x in batch]

        arguments = [
            {**problem_functions, "x": x, "step_id": step}
            for x, step in zip(starts, scheduled_steps, strict=False)
        ]

        batch_results = batch_evaluator(
            func=local_algorithm,
            arguments=arguments,
            unpack_symbol="**",
            n_cores=options.n_cores,
            error_handling=options.optimization_error_handling,
        )

        state, is_converged = update_convergence_state(
            current_state=state,
            starts=starts,
            results=batch_results,
            convergence_criteria=convergence_criteria,
            primary_key=primary_key,
        )
        opt_counter += len(batch)
        scheduled_steps = scheduled_steps[len(batch) :]
        if is_converged:
            if logging:
                for step in scheduled_steps:
                    update_step_status(
                        step=step,
                        new_status="skipped",
                        database=database,
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
        seed (int | np.random.Generator | None): Random seed.

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

    for name, bound in zip(["lower", "upper"], [lower, upper], strict=False):
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
        rng = get_rng(seed)
        sample_unscaled = rng.uniform(size=(n_samples, len(lower)))

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

    arguments = [{"x": x, "fixed_log_data": {"step": int(step_id)}} for x in sample]

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


def update_convergence_state(
    current_state, starts, results, convergence_criteria, primary_key
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
        primary_key: The primary criterion entry of the local optimizer. Needed to
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
    valid_new_x = [res["solution_x"] for res in valid_results]
    valid_new_y = []

    # make the criterion output scalar if a least squares optimizer returns an
    # array as solution_criterion.
    for res in valid_results:
        if np.isscalar(res["solution_criterion"]):
            valid_new_y.append(res["solution_criterion"])
        else:
            valid_new_y.append(
                aggregate_func_output_to_value(
                    f_eval=res["solution_criterion"],
                    primary_key=primary_key,
                )
            )

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
