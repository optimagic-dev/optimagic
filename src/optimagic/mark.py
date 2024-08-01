from enum import Enum
from typing import Callable, TypeVar

from optimagic.typing import FunctionValue, PyTree

ScalarFuncT = TypeVar("ScalarFuncT", bound=Callable[..., float | FunctionValue])
VectorFuncT = TypeVar("VectorFuncT", bound=Callable[..., PyTree | FunctionValue])


class ProblemType(Enum):
    """Enum class for function types."""

    SCALAR = "scalar"
    LEAST_SQUARES = "least_squares"
    LIKELIHOOD = "likelihood"


def scalar(func: ScalarFuncT) -> ScalarFuncT:
    """Mark a function as a scalar function."""
    func._problem_type = ProblemType.SCALAR  # type: ignore
    return func


def least_squares(func: VectorFuncT) -> VectorFuncT:
    """Mark a function as a least squares function."""
    func._problem_type = ProblemType.LEAST_SQUARES  # type: ignore
    return func


def likelihood(func: VectorFuncT) -> VectorFuncT:
    """Mark a function as a likelihood function."""
    func._problem_type = ProblemType.LIKELIHOOD  # type: ignore
    return func
