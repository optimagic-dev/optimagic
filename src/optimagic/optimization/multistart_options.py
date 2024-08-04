from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Sequence, TypedDict, get_args

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.exceptions import InvalidMultistartError
from optimagic.typing import PyTree

# ======================================================================================
# Public Options
# ======================================================================================

MultistartSamplingMethod = Literal["sobol", "random", "halton", "latin_hypercube"]
MultistartMixingWeightMethod = Literal["tiktak", "linear"]
MultistartSamplingDistribution = Literal["uniform", "triangular"]


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
        error_handling: The error handling for exploration and optimization errors.
            Allowed are "raise" and "continue".

    Raises:
        InvalidMultistartError: If the multistart options cannot be processed, e.g.
            because they do not have the correct type.

    """

    n_samples: int | None = None
    n_optimizations: int | None = None
    sampling_distribution: MultistartSamplingDistribution = "uniform"
    sampling_method: MultistartSamplingMethod = "random"
    sample: Sequence[PyTree] | None = None
    mixing_weight_method: (
        MultistartMixingWeightMethod | Callable[[int, int, float, float], float]
    ) = "tiktak"
    mixing_weight_bounds: tuple[float, float] = (0.1, 0.995)
    convergence_relative_params_tolerance: float = 0.01
    convergence_max_discoveries: int = 2
    n_cores: int = 1
    # TODO: Add more informative type hint for batch_evaluator
    batch_evaluator: Literal["joblib", "pathos"] | Callable = "joblib"  # type: ignore
    batch_size: int | None = None
    seed: int | np.random.Generator | None = None
    error_handling: Literal["raise", "continue"] = "continue"

    def __post_init__(self) -> None:
        _validate_attribute_types_and_values(self)


class MultistartOptionsDict(TypedDict):
    n_samples: NotRequired[int | None]
    n_optimizations: NotRequired[int | None]
    sampling_distribution: NotRequired[MultistartSamplingDistribution]
    sampling_method: NotRequired[MultistartSamplingMethod]
    sample: NotRequired[Sequence[PyTree] | None]
    mixing_weight_method: NotRequired[
        MultistartMixingWeightMethod | Callable[[int, int, float, float], float]
    ]
    mixing_weight_bounds: NotRequired[tuple[float, float]]
    convergence_relative_params_tolerance: NotRequired[float]
    convergence_max_discoveries: NotRequired[int]
    n_cores: NotRequired[int]
    batch_evaluator: NotRequired[Literal["joblib", "pathos"] | Callable]  # type: ignore
    batch_size: NotRequired[int | None]
    seed: NotRequired[int | np.random.Generator | None]
    error_handling: NotRequired[Literal["raise", "continue"]]


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

    if options.sampling_distribution not in get_args(MultistartSamplingDistribution):
        raise InvalidMultistartError(
            f"Invalid sampling distribution: {options.sampling_distribution}. Sampling "
            f"distribution must be one of {get_args(MultistartSamplingDistribution)}."
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
    ) and options.mixing_weight_method not in get_args(MultistartMixingWeightMethod):
        raise InvalidMultistartError(
            f"Invalid mixing weight method: {options.mixing_weight_method}. Mixing "
            "weight method must be Callable or one of "
            f"{get_args(MultistartMixingWeightMethod)}."
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

    if options.error_handling not in ("raise", "continue"):
        raise InvalidMultistartError(
            f"Invalid error handling: {options.error_handling}. Error handling must be "
            "'raise' or 'continue'."
        )


# ======================================================================================
# Internal Options
# ======================================================================================


def _tiktak_weights(
    iteration: int, n_iterations: int, min_weight: float, max_weight: float
) -> float:
    return np.clip(np.sqrt(iteration / n_iterations), min_weight, max_weight)


def _linear_weights(
    iteration: int, n_iterations: int, min_weight: float, max_weight: float
) -> float:
    unscaled = iteration / n_iterations
    span = max_weight - min_weight
    return min_weight + unscaled * span


WEIGHT_FUNCTIONS = {
    "tiktak": _tiktak_weights,
    "linear": _linear_weights,
}


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
    weight_func: Callable[[int, int], float]
    convergence_relative_params_tolerance: float
    convergence_max_discoveries: int
    n_cores: int
    # TODO: Add more informative type hint for batch_evaluator
    batch_evaluator: Callable  # type: ignore
    batch_size: int
    seed: int | np.random.Generator | None
    error_handling: Literal["raise", "continue"]
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

    if callable(options.mixing_weight_method):
        weight_func = options.mixing_weight_method
    else:
        _weight_method = WEIGHT_FUNCTIONS[options.mixing_weight_method]

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
        error_handling=options.error_handling,
        # Updated attributes
        n_samples=n_samples,
        sample=sample,
        weight_func=weight_func,
        n_optimizations=n_optimizations,
        batch_evaluator=batch_evaluator,
        batch_size=batch_size,
    )
