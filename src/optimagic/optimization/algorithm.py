import functools
import typing
import warnings
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, TypeVar

import numpy as np
import pydantic
from numpy.typing import NDArray
from typing_extensions import Self

from optimagic.exceptions import InvalidAlgoInfoError, InvalidAlgoOptionError
from optimagic.logging.types import StepStatus
from optimagic.optimization.history import History
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel

DataclassT = TypeVar("DataclassT")

OPTION_VALIDATION_CONFIG = pydantic.ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
    validate_default=True,
)
"""Pydantic config for user-facing options: coerce generous inputs to strict types."""

STRICT_VALIDATION_CONFIG = pydantic.ConfigDict(
    strict=True,
    arbitrary_types_allowed=True,
    extra="forbid",
    validate_default=True,
)
"""Pydantic config for internal types: reject inputs that need conversion."""


def validated_dataclass(
    config: pydantic.ConfigDict,
    make_error: Callable[[pydantic.ValidationError], Exception],
) -> Callable[[type[DataclassT]], type[DataclassT]]:
    """Create a class decorator that adds pydantic validation to a frozen dataclass.

    The decorated class is re-created as a pydantic dataclass, so field values are
    validated and converted according to their type annotations on every
    instantiation (including via ``dataclasses.replace``). Annotations are resolved
    at runtime, so this also works in modules using
    ``from __future__ import annotations``.

    Args:
        config: The pydantic config that controls validation behavior.
        make_error: Called with the raised ``pydantic.ValidationError`` to build the
            exception that is raised in its place.

    Returns:
        A class decorator for frozen dataclasses.

    """

    def decorator(cls: type[DataclassT]) -> type[DataclassT]:
        out = pydantic.dataclasses.dataclass(frozen=True, config=config)(cls)
        # pydantic re-creates the class, which loses attributes that tooling and
        # introspection rely on.
        out.__doc__ = cls.__doc__
        out.__annotations__ = dict(cls.__annotations__)
        original_init = out.__init__

        @functools.wraps(original_init)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            try:
                original_init(self, *args, **kwargs)
            except pydantic.ValidationError as e:
                raise make_error(e) from e

        out.__init__ = __init__  # type: ignore[method-assign]
        return typing.cast("type[DataclassT]", out)

    return decorator


def _algo_info_error(e: pydantic.ValidationError) -> Exception:
    msg = f"The following arguments to AlgoInfo or `mark.minimizer` are invalid:\n\n{e}"
    return InvalidAlgoInfoError(msg)


def _internal_optimize_result_error(e: pydantic.ValidationError) -> Exception:
    msg = f"The following arguments to InternalOptimizeResult are invalid:\n\n{e}"
    return TypeError(msg)


@validated_dataclass(config=STRICT_VALIDATION_CONFIG, make_error=_algo_info_error)
@dataclass(frozen=True)
class AlgoInfo:
    name: str
    solver_type: AggregationLevel
    is_available: bool
    is_global: bool
    needs_jac: bool
    needs_hess: bool
    needs_bounds: bool
    supports_parallelism: bool
    supports_bounds: bool
    supports_infinite_bounds: bool
    supports_linear_constraints: bool
    supports_nonlinear_constraints: bool
    disable_history: bool = False
    experimental: bool = False


@validated_dataclass(
    config=STRICT_VALIDATION_CONFIG, make_error=_internal_optimize_result_error
)
@dataclass(frozen=True)
class InternalOptimizeResult:
    """Internal representation of the result of an optimization problem.

    Args:
        x: The optimal parameters.
        fun: The function value at the optimal parameters.
        success: Whether the optimization was successful.
        message: A message from the optimizer.
        status: The status of the optimization.
        n_fun_evals: The number of function evaluations.
        n_jac_evals: The number of gradient or jacobian evaluations.
        n_hess_evals: The number of Hessian evaluations.
        n_iterations: The number of iterations.
        jac: The Jacobian of the objective function at the optimal parameters.
        hess: The Hessian of the objective function at the optimal parameters.
        hess_inv: The inverse of the Hessian of the objective function at the optimal
            parameters.
        max_constraint_violation: The maximum constraint violation.
        info: Additional information from the optimizer.

    """

    x: NDArray[np.float64]
    fun: float | NDArray[np.float64]
    success: bool | None = None
    message: str | None = None
    status: int | None = None
    n_fun_evals: int | None = None
    n_jac_evals: int | None = None
    n_hess_evals: int | None = None
    n_iterations: int | None = None
    jac: NDArray[np.float64] | None = None
    hess: NDArray[np.float64] | None = None
    hess_inv: NDArray[np.float64] | None = None
    max_constraint_violation: float | None = None
    info: dict[str, typing.Any] | None = None
    history: History | None = None
    multistart_info: dict[str, typing.Any] | None = None


class AlgorithmMeta(ABCMeta):
    """Metaclass to get repr, algo_info and name for classes, not just instances."""

    def __repr__(self) -> str:
        if hasattr(self, "__algo_info__") and self.__algo_info__ is not None:
            out = f"om.algos.{self.__algo_info__.name}"
        else:
            out = self.__class__.__name__
        return out

    @property
    def name(self) -> str:
        if hasattr(self, "__algo_info__") and self.__algo_info__ is not None:
            out = self.__algo_info__.name
        else:
            out = self.__class__.__name__
        return out

    @property
    def algo_info(self) -> AlgoInfo:
        if not hasattr(self, "__algo_info__") or self.__algo_info__ is None:
            msg = (
                f"The algorithm {self.name} does not have have the __algo_info__ "
                "attribute. Use the `mark.minimizer` decorator to add this attribute."
            )
            raise AttributeError(msg)

        return self.__algo_info__


@dataclass(frozen=True)
class Algorithm(ABC, metaclass=AlgorithmMeta):
    """Base class for all optimization algorithms in optimagic.

    To add an optimizer to optimagic you need to subclass Algorithm and overide the
    ``_solve_internal_problem`` method.

    """

    @abstractmethod
    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        pass

    def with_option(self, **kwargs: Any) -> Self:
        """Create a modified copy with the given options."""
        valid_keys = set(self.__dataclass_fields__) - {"__algo_info__"}
        invalid = set(kwargs) - valid_keys
        if invalid:
            raise InvalidAlgoOptionError(
                f"The keyword arguments {invalid} are not valid options for "
                f"the algorithm {self.name}"
            )
        return replace(self, **kwargs)

    def with_stopping(self, **kwargs: Any) -> Self:
        """Create a modified copy with the given stopping options."""
        options = {}
        for k, v in kwargs.items():
            if k.startswith("stopping_"):
                options[k] = v
            else:
                options[f"stopping_{k}"] = v

        return self.with_option(**options)

    def with_convergence(self, **kwargs: Any) -> Self:
        """Create a modified copy with the given convergence options."""
        options = {}
        for k, v in kwargs.items():
            if k.startswith("convergence_"):
                options[k] = v
            else:
                options[f"convergence_{k}"] = v

        return self.with_option(**options)

    def solve_internal_problem(
        self,
        problem: InternalOptimizationProblem,
        x0: NDArray[np.float64],
        step_id: int,
    ) -> InternalOptimizeResult:
        """Solve the internal optimization problem.

        This method is called internally by `minimize` or `maximize` to solve the
        internal optimization problem and process the results.

        """
        problem = problem.with_new_history().with_step_id(step_id)

        if problem.logger:
            problem.logger.step_store.update(
                step_id, {"status": str(StepStatus.RUNNING.value)}
            )

        result = self._solve_internal_problem(problem, x0)

        if (not self.algo_info.disable_history) and (result.history is None):
            result = replace(result, history=problem.history)

        if problem.logger:
            problem.logger.step_store.update(
                step_id, {"status": str(StepStatus.COMPLETE.value)}
            )

        return result

    def with_option_if_applicable(self, **kwargs: Any) -> Self:
        """Call with_option only with applicable keyword arguments."""
        valid_keys = set(self.__dataclass_fields__) - {"__algo_info__"}
        invalid = set(kwargs) - valid_keys
        if invalid:
            msg = (
                "The following algo_options were ignored because they are not "
                f"compatible with {self.name}:\n\n {invalid}"
            )
            warnings.warn(msg)

        kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return self.with_option(**kwargs)

    @property
    def name(self) -> str:
        """The name of the algorithm."""
        # cannot call algo_info here because it would be an infinite recursion
        if hasattr(self, "__algo_info__") and self.__algo_info__ is not None:
            return self.__algo_info__.name
        return self.__class__.__name__

    @property
    def algo_info(self) -> AlgoInfo:
        """Information about the algorithm."""
        if not hasattr(self, "__algo_info__") or self.__algo_info__ is None:
            msg = (
                f"The algorithm {self.name} does not have have the __algo_info__ "
                "attribute. Use the `mark.minimizer` decorator to add this attribute."
            )
            raise AttributeError(msg)

        return self.__algo_info__
