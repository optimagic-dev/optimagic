from dataclasses import dataclass
from typing import Callable, Literal, TypedDict

from typing_extensions import NotRequired

from optimagic.config import DEFAULT_N_CORES
from optimagic.exceptions import InvalidNumdiffError


@dataclass(frozen=True)
class NumdiffOptions:
    """Options for numerical differentiation.

    Attributes:
        method: The method to use for numerical differentiation. Can be "central",
            "forward", or "backward".
        step_size: The step size to use for numerical differentiation. If None, the
            default step size will be used.
        scaling_factor: The scaling factor to use for numerical differentiation.
        min_steps: The minimum step size to use for numerical differentiation. If None,
            the default minimum step size will be used.
        n_cores: The number of cores to use for numerical differentiation.
        batch_evaluator: The batch evaluator to use for numerical differentiation. Can
            be "joblib" or "pathos", or a custom function.

    Raises:
        InvalidNumdiffError: If the numdiff options cannot be processed, e.g. because
            they do not have the correct type.

    """

    method: Literal["central", "forward", "backward"] = "central"
    step_size: float | None = None
    scaling_factor: float = 1
    min_steps: float | None = None
    n_cores: int = DEFAULT_N_CORES
    batch_evaluator: Literal["joblib", "pathos"] | Callable = "joblib"  # type: ignore

    def __post_init__(self) -> None:
        _validate_attribute_types_and_values(self)


class NumdiffOptionsDict(TypedDict):
    method: NotRequired[Literal["central", "forward", "backward"]]
    step_size: NotRequired[float | None]
    scaling_factor: NotRequired[float]
    min_steps: NotRequired[float | None]
    n_cores: NotRequired[int]
    batch_evaluator: NotRequired[Literal["joblib", "pathos"] | Callable]  # type: ignore


def pre_process_numdiff_options(
    numdiff_options: NumdiffOptions | NumdiffOptionsDict,
) -> NumdiffOptions | None:
    """Convert all valid types of Numdiff options to optimagic.NumdiffOptions class.

    This just harmonizes multiple ways of specifying numdiff options into a single
    format. It performs runtime type checks, but it does not check whether numdiff
    options are consistent with other option choices.

    Args:
        numdiff_options: The user provided numdiff options.

    Returns:
        The numdiff options in the optimagic format.

    Raises:
        InvalidNumdiffOptionsError: If numdiff options cannot be processed, e.g. because
            they do not have the correct type.

    """
    if isinstance(numdiff_options, NumdiffOptions) or numdiff_options is None:
        pass
    else:
        try:
            numdiff_options = NumdiffOptions(**numdiff_options)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if isinstance(e, InvalidNumdiffError):
                raise e
            raise InvalidNumdiffError(
                f"Invalid numdiff options of type: {type(numdiff_options)}. Numdiff "
                "options must be of type optimagic.NumdiffOptions, a dictionary with a"
                "subset of the keys {'method', 'step_size', 'scaling_factor', "
                "'min_steps', 'n_cores', 'batch_evaluator'}, or None."
            ) from e

    return numdiff_options


def _validate_attribute_types_and_values(options: NumdiffOptions) -> None:
    if options.method not in {"central", "forward", "backward"}:
        raise InvalidNumdiffError(
            f"Invalid numdiff `method`: {options.method}. Numdiff `method` must be "
            "one of 'central', 'forward', or 'backward'."
        )

    if options.step_size is not None and (
        not isinstance(options.step_size, int | float) or options.step_size <= 0
    ):
        raise InvalidNumdiffError(
            f"Invalid numdiff `step_size`: {options.step_size}. Step size must be an "
            "integer or float greater than 0."
        )

    if (
        not isinstance(options.scaling_factor, int | float)
        or options.scaling_factor <= 0
    ):
        raise InvalidNumdiffError(
            f"Invalid numdiff `scaling_factor`: {options.scaling_factor}. Scaling "
            "factor must be an integer or float greater than 0."
        )

    if options.min_steps is not None and (
        not isinstance(options.min_steps, int | float) or options.min_steps <= 0
    ):
        raise InvalidNumdiffError(
            f"Invalid numdiff `min_steps`: {options.min_steps}. Minimum step "
            "size must be an integer or float greater than 0."
        )

    if not isinstance(options.n_cores, int) or options.n_cores <= 0:
        raise InvalidNumdiffError(
            f"Invalid numdiff `n_cores`: {options.n_cores}. Number of cores "
            "must be an integer greater than 0."
        )

    if not callable(options.batch_evaluator) and options.batch_evaluator not in {
        "joblib",
        "pathos",
    }:
        raise InvalidNumdiffError(
            f"Invalid numdiff `batch_evaluator`: {options.batch_evaluator}. Batch "
            "evaluator must be a callable or one of 'joblib', 'pathos'."
        )
