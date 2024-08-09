from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from optimagic.typing import AggregationLevel

P = ParamSpec("P")


ScalarFuncT = TypeVar("ScalarFuncT", bound=Callable[..., Any])
VectorFuncT = TypeVar("VectorFuncT", bound=Callable[..., Any])


def scalar(func: ScalarFuncT) -> ScalarFuncT:
    """Mark a function as a scalar function."""
    wrapper = func
    try:
        wrapper._problem_type = AggregationLevel.SCALAR  # type: ignore
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:

        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore
            return func(*args, **kwargs)

        wrapper._problem_type = AggregationLevel.SCALAR  # type: ignore
    return wrapper


def least_squares(func: VectorFuncT) -> VectorFuncT:
    """Mark a function as a least squares function."""
    wrapper = func
    try:
        wrapper._problem_type = AggregationLevel.LEAST_SQUARES  # type: ignore
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:

        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore
            return func(*args, **kwargs)

        wrapper._problem_type = AggregationLevel.LEAST_SQUARES  # type: ignore
    return wrapper


def likelihood(func: VectorFuncT) -> VectorFuncT:
    """Mark a function as a likelihood function."""
    wrapper = func
    try:
        wrapper._problem_type = AggregationLevel.LIKELIHOOD  # type: ignore
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:

        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore
            return func(*args, **kwargs)

        wrapper._problem_type = AggregationLevel.LIKELIHOOD  # type: ignore
    return wrapper
