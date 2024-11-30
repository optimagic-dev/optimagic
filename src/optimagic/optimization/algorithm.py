import typing
import warnings
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from optimagic.exceptions import InvalidAlgoInfoError, InvalidAlgoOptionError
from optimagic.logging.types import StepStatus
from optimagic.optimization.convergence_report import get_convergence_report
from optimagic.optimization.history import History
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.parameters.conversion import Converter
from optimagic.type_conversion import TYPE_CONVERTERS
from optimagic.typing import AggregationLevel, Direction, ExtraResultFields
from optimagic.utilities import isscalar


@dataclass(frozen=True)
class AlgoInfo:
    name: str
    solver_type: AggregationLevel
    is_available: bool
    is_global: bool
    needs_jac: bool
    needs_hess: bool
    supports_parallelism: bool
    supports_bounds: bool
    supports_linear_constraints: bool
    supports_nonlinear_constraints: bool
    disable_history: bool = False

    def __post_init__(self) -> None:
        report: list[str] = []
        if not isinstance(self.name, str):
            report.append("name must be a string")
        if not isinstance(self.solver_type, AggregationLevel):
            report.append("problem_type must be an AggregationLevel")
        if not isinstance(self.is_available, bool):
            report.append("is_available must be a bool")
        if not isinstance(self.is_global, bool):
            report.append("is_global must be a bool")
        if not isinstance(self.needs_jac, bool):
            report.append("needs_jac must be a bool")
        if not isinstance(self.needs_hess, bool):
            report.append("needs_hess must be a bool")
        if not isinstance(self.supports_parallelism, bool):
            report.append("supports_parallelism must be a bool")
        if not isinstance(self.supports_bounds, bool):
            report.append("supports_bounds must be a bool")
        if not isinstance(self.supports_linear_constraints, bool):
            report.append("supports_linear_constraints must be a bool")
        if not isinstance(self.supports_nonlinear_constraints, bool):
            report.append("supports_nonlinear_constraints must be a bool")
        if not isinstance(self.disable_history, bool):
            report.append("disable_history must be a bool")

        if report:
            msg = (
                "The following arguments to AlgoInfo or `mark.minimizer` are "
                "invalid:\n" + "\n".join(report)
            )
            raise InvalidAlgoInfoError(msg)


@dataclass(frozen=True)
class InternalOptimizeResult:
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

    def __post_init__(self) -> None:
        report: list[str] = []
        if not isinstance(self.x, np.ndarray):
            report.append("x must be a numpy array")

        if not (isinstance(self.fun, np.ndarray) or np.isscalar(self.fun)):
            report.append("fun must be a numpy array or scalar")

        if self.success is not None and not isinstance(self.success, bool):
            report.append("success must be a bool or None")

        if self.message is not None and not isinstance(self.message, str):
            report.append("message must be a string or None")

        if self.n_fun_evals is not None and not isinstance(self.n_fun_evals, int):
            report.append("n_fun_evals must be an int or None")

        if self.n_jac_evals is not None and not isinstance(self.n_jac_evals, int):
            report.append("n_jac_evals must be an int or None")

        if self.n_hess_evals is not None and not isinstance(self.n_hess_evals, int):
            report.append("n_hess_evals must be an int or None")

        if self.n_iterations is not None and not isinstance(self.n_iterations, int):
            report.append("n_iterations must be an int or None")

        if self.jac is not None and not isinstance(self.jac, np.ndarray):
            report.append("jac must be a numpy array or None")

        if self.hess is not None and not isinstance(self.hess, np.ndarray):
            report.append("hess must be a numpy array or None")

        if self.hess_inv is not None and not isinstance(self.hess_inv, np.ndarray):
            report.append("hess_inv must be a numpy array or None")

        if self.max_constraint_violation is not None and not np.isscalar(
            self.max_constraint_violation
        ):
            report.append("max_constraint_violation must be a scalar or None")

        if self.info is not None and not isinstance(self.info, dict):
            report.append("info must be a dictionary or None")

        if self.status is not None and not isinstance(self.status, int):
            report.append("status must be an int or None")

        if self.max_constraint_violation and not isinstance(
            self.max_constraint_violation, float
        ):
            report.append("max_constraint_violation must be a float or None")

        if report:
            msg = (
                "The following arguments to InternalOptimizeResult are invalid:\n"
                + "\n".join(report)
            )
            raise TypeError(msg)

    def create_optimize_result(
        self,
        converter: Converter,
        solver_type: AggregationLevel,
        extra_fields: ExtraResultFields,
    ) -> OptimizeResult:
        """Process an internal optimizer result."""
        params = converter.params_from_internal(self.x)
        if isscalar(self.fun):
            fun = float(self.fun)
        elif solver_type == AggregationLevel.LIKELIHOOD:
            fun = float(np.sum(self.fun))
        elif solver_type == AggregationLevel.LEAST_SQUARES:
            fun = np.dot(self.fun, self.fun)

        if extra_fields.direction == Direction.MAXIMIZE:
            fun = -fun

        if self.history is not None:
            conv_report = get_convergence_report(
                history=self.history, direction=extra_fields.direction
            )
        else:
            conv_report = None

        out = OptimizeResult(
            params=params,
            fun=fun,
            start_fun=extra_fields.start_fun,
            start_params=extra_fields.start_params,
            algorithm=extra_fields.algorithm,
            direction=extra_fields.direction.value,
            n_free=extra_fields.n_free,
            message=self.message,
            success=self.success,
            n_fun_evals=self.n_fun_evals,
            n_jac_evals=self.n_jac_evals,
            n_hess_evals=self.n_hess_evals,
            n_iterations=self.n_iterations,
            status=self.status,
            jac=self.jac,
            hess=self.hess,
            hess_inv=self.hess_inv,
            max_constraint_violation=self.max_constraint_violation,
            history=self.history,
            algorithm_output=self.info,
            convergence_report=conv_report,
        )
        return out


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
    @abstractmethod
    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        pass

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            raw_value = getattr(self, field)
            target_type = typing.cast(type, self.__dataclass_fields__[field].type)
            if target_type in TYPE_CONVERTERS:
                try:
                    value = TYPE_CONVERTERS[target_type](raw_value)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    msg = (
                        f"Could not convert the value of the field {field} to the "
                        f"expected type {target_type}."
                    )
                    raise InvalidAlgoOptionError(msg) from e

                object.__setattr__(self, field, value)

    def with_option(self, **kwargs: Any) -> Self:
        valid_keys = set(self.__dataclass_fields__) - {"__algo_info__"}
        invalid = set(kwargs) - valid_keys
        if invalid:
            raise InvalidAlgoOptionError(
                f"The keyword arguments {invalid} are not valid options for "
                f"the algorithm {self.name}"
            )
        return replace(self, **kwargs)

    def with_stopping(self, **kwargs: Any) -> Self:
        options = {}
        for k, v in kwargs.items():
            if k.startswith("stopping_"):
                options[k] = v
            else:
                options[f"stopping_{k}"] = v

        return self.with_option(**options)

    def with_convergence(self, **kwargs: Any) -> Self:
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
    ) -> OptimizeResult:
        problem = problem.with_new_history().with_step_id(step_id)

        if problem.logger:
            problem.logger.step_store.update(
                step_id, {"status": str(StepStatus.RUNNING.value)}
            )

        raw_res = self._solve_internal_problem(problem, x0)

        if (not self.algo_info.disable_history) and (raw_res.history is None):
            raw_res = replace(raw_res, history=problem.history)

        if problem.logger:
            problem.logger.step_store.update(
                step_id, {"status": str(StepStatus.COMPLETE.value)}
            )

        res = raw_res.create_optimize_result(
            converter=problem.converter,
            solver_type=self.algo_info.solver_type,
            extra_fields=problem.static_result_fields,
        )

        return res

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
        # cannot call algo_info here because it would be an infinite recursion
        if hasattr(self, "__algo_info__") and self.__algo_info__ is not None:
            return self.__algo_info__.name
        return self.__class__.__name__

    @property
    def algo_info(self) -> AlgoInfo:
        if not hasattr(self, "__algo_info__") or self.__algo_info__ is None:
            msg = (
                f"The algorithm {self.name} does not have have the __algo_info__ "
                "attribute. Use the `mark.minimizer` decorator to add this attribute."
            )
            raise AttributeError(msg)

        return self.__algo_info__
