from dataclasses import dataclass
from typing import Any, Callable

PyTree = Any
PyTreeRegistry = dict[type | str, dict[str, Callable[[Any], Any]]]


@dataclass(frozen=True)
class FunctionValue:
    value: float | PyTree
    info: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScalarFunctionValue(FunctionValue):
    value: float
    info: dict[str, Any] | None = None


@dataclass(frozen=True)
class LeastSquaresFunctionValue(FunctionValue):
    value: PyTree
    info: dict[str, Any] | None = None


@dataclass(frozen=True)
class LikelihoodFunctionValue(FunctionValue):
    value: PyTree
    info: dict[str, Any] | None = None
