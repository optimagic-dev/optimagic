from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from optimagic.optimization.algorithm import AlgoInfo
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


# TODO: I get an error when adding bound=Algorithm to AlgorithmSubclass. Why?
AlgorithmSubclass = TypeVar("AlgorithmSubclass")


def minimizer(
    name: str,
    solver_type: AggregationLevel,
    is_available: bool,
    is_global: bool,
    needs_jac: bool,
    needs_hess: bool,
    supports_parallelism: bool,
    supports_bounds: bool,
    supports_linear_constraints: bool,
    supports_nonlinear_constraints: bool,
    disable_history: bool = False,
) -> Callable[[AlgorithmSubclass], AlgorithmSubclass]:
    """Mark an algorithm as a optimagic minimizer and add AlgoInfo.

    Args:
        name: The name of the algorithm as a string. Used in error messages, warnings
            and the OptimizeResult.
        problem_type: The type of optimization problem the algorithm solves. Used to
            distinguish between scalar, least-squares and likelihood optimizers.

    """

    def decorator(cls: AlgorithmSubclass) -> AlgorithmSubclass:
        algo_info = AlgoInfo(
            name=name,
            solver_type=solver_type,
            is_available=is_available,
            is_global=is_global,
            needs_jac=needs_jac,
            needs_hess=needs_hess,
            supports_parallelism=supports_parallelism,
            supports_bounds=supports_bounds,
            supports_linear_constraints=supports_linear_constraints,
            supports_nonlinear_constraints=supports_nonlinear_constraints,
            disable_history=disable_history,
        )
        cls.__algo_info__ = algo_info  # type: ignore
        return cls

    return decorator
