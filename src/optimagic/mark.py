from typing import Any, Callable, TypeVar

from optimagic.typing import AggregationLevel

ScalarFuncT = TypeVar("ScalarFuncT", bound=Callable[..., Any])
VectorFuncT = TypeVar("VectorFuncT", bound=Callable[..., Any])


def scalar(func: ScalarFuncT) -> ScalarFuncT:
    """Mark a function as a scalar function."""
    func._problem_type = AggregationLevel.SCALAR  # type: ignore
    return func


def least_squares(func: VectorFuncT) -> VectorFuncT:
    """Mark a function as a least squares function."""
    func._problem_type = AggregationLevel.LEAST_SQUARES  # type: ignore
    return func


def likelihood(func: VectorFuncT) -> VectorFuncT:
    """Mark a function as a likelihood function."""
    func._problem_type = AggregationLevel.LIKELIHOOD  # type: ignore
    return func
