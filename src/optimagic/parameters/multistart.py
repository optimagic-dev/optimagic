from dataclasses import dataclass
from typing import Callable, Literal, Sequence, TypedDict, get_args

import numpy as np
from typing_extensions import NotRequired

from optimagic.exceptions import InvalidMultistartError
from optimagic.typing import PyTree

MultistartSamplingMethod = Literal[
    "sobol", "random", "halton", "hammersley", "korobov", "latin_hypercube"
]


@dataclass
class MultistartOptions:
    """Multistart options in optimization problems.

    A description of the attributes will be written once the multistart code is
    refactored.

    """

    n_samples: int | None = None
    share_optimizations: float = 0.1
    sampling_distribution: Literal["uniform", "triangular"] = "uniform"
    sampling_method: MultistartSamplingMethod = "sobol"
    sample: Sequence[PyTree] | None = None
    mixing_weight_method: Literal["tiktak", "linear"] = "tiktak"
    mixing_weight_bounds: tuple[float, float] = (0.1, 0.995)
    convergence_relative_params_tolerance: float = 0.01
    convergence_max_discoveries: int = 2
    n_cores: int = 1
    batch_evaluator: Literal["joblib", "pathos"] | Callable = "joblib"  # type: ignore
    batch_size: int | None = None
    seed: int | np.random.Generator | None = None
    exploration_error_handling: Literal["raise", "continue"] = "continue"
    optimization_error_handling: Literal["raise", "continue"] = "continue"
    n_optimizations: int | None = None


class MultistartOptionsDict(TypedDict):
    n_samples: NotRequired[int | None]
    share_optimizations: NotRequired[float]
    sampling_distribution: NotRequired[Literal["uniform", "triangular"]]
    sampling_method: NotRequired[MultistartSamplingMethod]
    sample: NotRequired[Sequence[PyTree] | None]
    mixing_weight_method: NotRequired[Literal["tiktak", "linear"]]
    mixing_weight_bounds: NotRequired[tuple[float, float]]
    convergence_relative_params_tolerance: NotRequired[float]
    convergence_max_discoveries: NotRequired[int]
    n_cores: NotRequired[int]
    batch_evaluator: NotRequired[Literal["joblib", "pathos"] | Callable]  # type: ignore
    batch_size: NotRequired[int | None]
    seed: NotRequired[int | np.random.Generator | None]
    exploration_error_handling: NotRequired[Literal["raise", "continue"]]
    optimization_error_handling: NotRequired[Literal["raise", "continue"]]
    n_optimizations: NotRequired[int | None]


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
            raise InvalidMultistartError(
                f"Invalid multistart options of type: {type(multistart)}. Multistart "
                "options must be of type optimagic.MultistartOptions, a dictionary "
                "with valid keys, None, or a boolean."
            ) from e

    if isinstance(multistart, MultistartOptions):
        _validate_attribute_types_and_values(multistart)

    return multistart


def _validate_attribute_types_and_values(options: MultistartOptions) -> None:
    if not isinstance(options.n_samples, int | None):
        raise InvalidMultistartError(
            f"Invalid number of samples: {options.n_samples}. Number of samples "
            "must be a positive integer."
        )

    if not isinstance(options.share_optimizations, int | float):
        raise InvalidMultistartError(
            f"Invalid share of optimizations: {options.share_optimizations}. Share of "
            "optimizations must be a float or int."
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
            f"Invalid sample: {options.sample}. Sample must be a sequence of PyTrees."
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
        or set(type(x) for x in options.mixing_weight_bounds) != {float}
    ):
        raise InvalidMultistartError(
            f"Invalid mixing weight bounds: {options.mixing_weight_bounds}. Mixing "
            "weight bounds must be a tuple of two floats."
        )

    if not isinstance(options.convergence_relative_params_tolerance, int | float):
        raise InvalidMultistartError(
            "Invalid relative params tolerance:"
            f"{options.convergence_relative_params_tolerance}. Relative params "
            "tolerance must be a number."
        )

    if (
        not isinstance(options.convergence_max_discoveries, int)
        or options.convergence_max_discoveries < 1
    ):
        raise InvalidMultistartError(
            f"Invalid max discoveries: {options.convergence_max_discoveries}. Max "
            "discoveries must be a positive integer."
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

    if not isinstance(options.batch_size, int | None):
        raise InvalidMultistartError(
            f"Invalid batch size: {options.batch_size}. Batch size "
            "must be a positive integer."
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
