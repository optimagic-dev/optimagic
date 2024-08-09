import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ParamSpec

import numpy as np
from numpy.typing import NDArray
from pybaum import tree_just_flatten

from optimagic.exceptions import InvalidFunctionError
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import AggregationLevel, PyTree, Scalar
from optimagic.utilities import isscalar


@dataclass(frozen=True)
class FunctionValue:
    value: float | PyTree
    info: dict[str, Any] | None = None


class SpecificFunctionValue(FunctionValue, ABC):
    @abstractmethod
    def internal_value(
        self, solver_type: AggregationLevel
    ) -> float | NDArray[np.float64]:
        pass


@dataclass(frozen=True)
class ScalarFunctionValue(SpecificFunctionValue):
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

    def internal_value(self, solver_type: AggregationLevel) -> float:
        if solver_type == AggregationLevel.SCALAR:
            val = float(self.value)
        else:
            raise InvalidFunctionError(
                f"You are using a {solver_type.value} optimizer but provided a "
                "scalar objective function."
            )
        return val


@dataclass(frozen=True)
class LeastSquaresFunctionValue(SpecificFunctionValue):
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
        self, solver_type: AggregationLevel
    ) -> float | NDArray[np.float64]:
        resid = _get_flat_value(self.value)

        val: float | NDArray[np.float64]

        if solver_type == AggregationLevel.LEAST_SQUARES:
            val = resid
        elif solver_type == AggregationLevel.LIKELIHOOD:
            val = resid**2
        else:
            val = float(resid @ resid)
        return val


@dataclass(frozen=True)
class LikelihoodFunctionValue(SpecificFunctionValue):
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
        self, solver_type: AggregationLevel
    ) -> float | NDArray[np.float64]:
        loglikes = _get_flat_value(self.value)

        val: float | NDArray[np.float64]

        if solver_type == AggregationLevel.LIKELIHOOD:
            val = loglikes
        elif solver_type == AggregationLevel.SCALAR:
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


def convert_fun_output_to_function_value(
    raw: Scalar | PyTree | FunctionValue, problem_type: AggregationLevel
) -> SpecificFunctionValue:
    out: FunctionValue
    if problem_type == AggregationLevel.SCALAR:
        out = _convert_output_to_scalar_function_value(raw)
    elif problem_type == AggregationLevel.LEAST_SQUARES:
        out = _convert_output_to_least_squares_function_value(raw)
    elif problem_type == AggregationLevel.LIKELIHOOD:
        out = _convert_output_to_likelihood_function_value(raw)
    return out


def _convert_output_to_scalar_function_value(
    raw: Scalar | FunctionValue,
) -> ScalarFunctionValue:
    if isinstance(raw, ScalarFunctionValue):
        out = raw
    elif isinstance(raw, FunctionValue):
        out = ScalarFunctionValue(value=raw.value, info=raw.info)
    else:
        out = ScalarFunctionValue(value=raw)
    return out


def _convert_output_to_least_squares_function_value(
    raw: PyTree | FunctionValue,
) -> LeastSquaresFunctionValue:
    if isinstance(raw, LeastSquaresFunctionValue):
        out = raw
    elif isinstance(raw, FunctionValue):
        out = LeastSquaresFunctionValue(value=raw.value, info=raw.info)
    else:
        out = LeastSquaresFunctionValue(value=raw)
    return out


def _convert_output_to_likelihood_function_value(
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


def enforce_return_type(
    problem_type: AggregationLevel,
) -> Callable[
    [Callable[P, Scalar | PyTree | FunctionValue]], Callable[P, SpecificFunctionValue]
]:
    """Enforce a strict return type for objective functions based on problem_type.

    This has no effect if the function already returns the strictest possible type for
    the problem_type but converts everything else to that type.

    """

    def decorator_enforce(
        func: Callable[P, Scalar | PyTree | FunctionValue],
    ) -> Callable[P, SpecificFunctionValue]:
        if problem_type == AggregationLevel.SCALAR:

            @functools.wraps(func)
            def wrapper_enforce(
                *args: P.args, **kwargs: P.kwargs
            ) -> ScalarFunctionValue:
                raw = func(*args, **kwargs)
                return _convert_output_to_scalar_function_value(raw)
        elif problem_type == AggregationLevel.LEAST_SQUARES:

            @functools.wraps(func)
            def wrapper_enforce(
                *args: P.args, **kwargs: P.kwargs
            ) -> LeastSquaresFunctionValue:
                raw = func(*args, **kwargs)
                return _convert_output_to_least_squares_function_value(raw)
        elif problem_type == AggregationLevel.LIKELIHOOD:

            @functools.wraps(func)
            def wrapper_enforce(
                *args: P.args, **kwargs: P.kwargs
            ) -> LikelihoodFunctionValue:
                raw = func(*args, **kwargs)
                return _convert_output_to_likelihood_function_value(raw)

        return wrapper_enforce

    return decorator_enforce


def enforce_return_type_with_jac(
    problem_type: AggregationLevel,
) -> Callable[
    [Callable[P, tuple[Scalar | PyTree | FunctionValue, PyTree]]],
    Callable[P, tuple[SpecificFunctionValue, PyTree]],
]:
    """Enforce a strict return type for fun_and_jac based on problem_type.

    This has no effect if the first return value of the function already has the
    strictest possible type for the problem_type but converts everything else to that
    type. The second return value stays unchanged.

    """

    def decorator_enforce(
        func: Callable[P, tuple[Scalar | PyTree | FunctionValue, PyTree]],
    ) -> Callable[P, tuple[SpecificFunctionValue, PyTree]]:
        if problem_type == AggregationLevel.SCALAR:

            @functools.wraps(func)
            def wrapper_enforce(
                *args: P.args, **kwargs: P.kwargs
            ) -> tuple[ScalarFunctionValue, PyTree]:
                raw = func(*args, **kwargs)
                return (_convert_output_to_scalar_function_value(raw[0]), raw[1])
        elif problem_type == AggregationLevel.LEAST_SQUARES:

            @functools.wraps(func)
            def wrapper_enforce(
                *args: P.args, **kwargs: P.kwargs
            ) -> tuple[LeastSquaresFunctionValue, PyTree]:
                raw = func(*args, **kwargs)
                return (_convert_output_to_least_squares_function_value(raw[0]), raw[1])
        elif problem_type == AggregationLevel.LIKELIHOOD:

            @functools.wraps(func)
            def wrapper_enforce(
                *args: P.args, **kwargs: P.kwargs
            ) -> tuple[LikelihoodFunctionValue, PyTree]:
                raw = func(*args, **kwargs)
                return (_convert_output_to_likelihood_function_value(raw[0]), raw[1])

        return wrapper_enforce

    return decorator_enforce
