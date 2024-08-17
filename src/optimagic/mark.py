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
        solver_type: The type of optimization problem the algorithm solves. Used to
            distinguish between scalar, least-squares and likelihood optimizers. Can
            take the values AggregationLevel.SCALAR, AggregationLevel.LEAST_SQUARES and
            AggregationLevel.LIKELIHOOD.
        is_available: Whether the algorithm is installed.
        is_global: Whether the algorithm is a global optimizer.
        needs_jac: Whether the algorithm needs some kind of first derivative. This needs
            to be True if the algorithm uses `jac` or `fun_and_jac`.
        needs_hess: Whether the algorithm needs some kind of second derivative. This
            is not yet implemented and will be False for all currently wrapped
            algorithms.
        supports_parallelism: Whether the algorithm supports parallelism. This needs to
            be True if the algorithm previously took `n_cores` and/or `batch_evaluator`
            as arguments.
        supports_bounds: Whether the algorithm supports bounds. This needs to be True
            if the algorithm previously took `lower_bounds` and/or `upper_bounds` as
            arguments.
        supports_linear_constraints: Whether the algorithm supports linear constraints.
            This is not yet implemented and will be False for all currently wrapped
            algorithms.
        supports_nonlinear_constraints: Whether the algorithm supports nonlinear
            constraints. This needs to be True if the algorithm previously took
            `nonlinear_constraints` as an argument.
        disable_history: Whether the algorithm should disable history collection.

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
