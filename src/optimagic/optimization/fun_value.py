import functools
from dataclasses import dataclass
from typing import Any, Callable, ParamSpec

import numpy as np
from numpy.typing import NDArray
from pybaum import tree_just_flatten

from optimagic.exceptions import InvalidFunctionError
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import OptimizerType, PyTree, Scalar
from optimagic.utilities import isscalar


@dataclass(frozen=True)
class FunctionValue:
    value: float | PyTree
    info: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScalarFunctionValue(FunctionValue):
    value: Scalar
    info: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isscalar(self.value):
            raise InvalidFunctionError(
                f"Scalar objective values need to be scalars, not: {type(self.value)}. "
                "If you meant to provide a scalar objective function, make sure it "
                "returns a scalar value. If you meant to provide a least_squares or "
                "likelihood function, use the mark.least_squares or mark.likelihood "
                "decorators."
            )

    def internal_value(self, optimizer_type: OptimizerType) -> float:
        if optimizer_type == OptimizerType.SCALAR:
            val = float(self.value)
        else:
            raise InvalidFunctionError(
                f"You are using a {optimizer_type.value} optimizer but provided a "
                "scalar objective function."
            )
        return val


@dataclass(frozen=True)
class LeastSquaresFunctionValue(FunctionValue):
    value: PyTree
    info: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isscalar(self.value):
            raise InvalidFunctionError(
                "Least squares objective values cannot be scalars. Your value has "
                f"scalar type: {type(self.value)}. If you meant to provide a least "
                "squares objective function, make sure it does not have a scalar value."
                " If you meant to provide a scalar function, use the mark.scalar "
                "decorator."
            )

    def internal_value(
        self, optimizer_type: OptimizerType
    ) -> float | NDArray[np.float64]:
        resid = _get_flat_value(self.value)

        val: float | NDArray[np.float64]

        if optimizer_type == OptimizerType.LEAST_SQUARES:
            val = resid
        elif optimizer_type == OptimizerType.LIKELIHOOD:
            val = resid**2
        else:
            val = float(resid @ resid)
        return val


@dataclass(frozen=True)
class LikelihoodFunctionValue(FunctionValue):
    value: PyTree
    info: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isscalar(self.value):
            raise InvalidFunctionError(
                "Likelihood objective values cannot be scalars. Your value has scalar "
                f"type: {type(self.value)}. If you meant to provide a likelihood "
                "objective function, make sure it does not have a scalar value. If you "
                "meant to provide a scalar function, use the mark.scalar decorator."
            )

    def internal_value(
        self, optimizer_type: OptimizerType
    ) -> float | NDArray[np.float64]:
        loglikes = _get_flat_value(self.value)

        val: float | NDArray[np.float64]

        if optimizer_type == OptimizerType.LIKELIHOOD:
            val = loglikes
        elif optimizer_type == OptimizerType.SCALAR:
            val = float(np.sum(loglikes))
        else:
            raise InvalidFunctionError(
                "You are using a least_squares optimizer but provided a "
                "likelihood objective function."
            )
        return val


def _get_flat_value(value: PyTree) -> NDArray[np.float64]:
    """Flatten a PyTree value to a 1d numpy array with multiple fast paths."""
    if isinstance(value, np.ndarray) and value.ndim == 1:
        flat = value
    elif isinstance(value, np.ndarray):
        flat = value.flatten()
    else:
        registry = get_registry(extended=True)
        flat = tree_just_flatten(value, registry=registry)

    flat_arr = np.asarray(flat, dtype=np.float64)
    return flat_arr


def convert_output_to_scalar_function_value(
    raw: Scalar | FunctionValue,
) -> ScalarFunctionValue:
    if isinstance(raw, ScalarFunctionValue):
        out = raw
    elif isinstance(raw, FunctionValue):
        out = ScalarFunctionValue(value=raw.value, info=raw.info)
    else:
        out = ScalarFunctionValue(value=raw)
    return out


def convert_output_to_least_squares_function_value(
    raw: PyTree | FunctionValue,
) -> LeastSquaresFunctionValue:
    if isinstance(raw, LeastSquaresFunctionValue):
        out = raw
    elif isinstance(raw, FunctionValue):
        out = LeastSquaresFunctionValue(value=raw.value, info=raw.info)
    else:
        out = LeastSquaresFunctionValue(value=raw)
    return out


def convert_output_to_likelihood_function_value(
    raw: PyTree | FunctionValue,
) -> LikelihoodFunctionValue:
    if isinstance(raw, LikelihoodFunctionValue):
        out = raw
    elif isinstance(raw, FunctionValue):
        out = LikelihoodFunctionValue(value=raw.value, info=raw.info)
    else:
        out = LikelihoodFunctionValue(value=raw)
    return out


P = ParamSpec("P")


def enforce_least_squares(
    func: Callable[P, PyTree | FunctionValue],
) -> Callable[P, LeastSquaresFunctionValue]:
    """Make valid least squares functions return a LeastSquaresFunctionValue.

    This has no effect if the function already returns a LeastSquaresFunctionValue but
    converts a PyTree or FunctionValue to a LeastSquaresFunctionValue.

    Whereas the mark.least_squares decorator works for objective functions and
    derivatives, this is only meant for objective functions.

    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> LeastSquaresFunctionValue:
        raw = func(*args, **kwargs)
        return convert_output_to_least_squares_function_value(raw)

    return wrapper


def enforce_likelihood(
    func: Callable[P, PyTree | FunctionValue],
) -> Callable[P, LikelihoodFunctionValue]:
    """Make valid likelihood functions return a LikelihoodFunctionValue.

    This has no effect if the function already returns a LikelihoodFunctionValue but
    converts a PyTree or FunctionValue to a LikelihoodFunctionValue.

    Whereas the mark.likelihood decorator works for objective functions and derivatives,
    this is only meant for objective functions.

    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> LikelihoodFunctionValue:
        raw = func(*args, **kwargs)
        return convert_output_to_likelihood_function_value(raw)

    return wrapper


def enforce_scalar(
    func: Callable[P, float | FunctionValue],
) -> Callable[P, ScalarFunctionValue]:
    """Make valid scalar objective functions return a ScalarFunctionValue.

    This has no effect if the function already returns a ScalarFunctionValue but
    converts a float or FunctionValue to a ScalarFunctionValue.

    Whereas the mark.scalar decorator works for objective functions and derivatives,
    this is only meant for objective functions.

    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> ScalarFunctionValue:
        raw = func(*args, **kwargs)
        return convert_output_to_scalar_function_value(raw)

    return wrapper
