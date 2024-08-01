from dataclasses import dataclass
from typing import Any, Callable

PyTree = Any
PyTreeRegistry = dict[type | str, dict[str, Callable[[Any], Any]]]
from optimagic.exceptions import InvalidFunctionError
from optimagic.utilities import isscalar


@dataclass(frozen=True)
class FunctionValue:
    value: float | PyTree
    info: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScalarFunctionValue(FunctionValue):
    value: float
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
