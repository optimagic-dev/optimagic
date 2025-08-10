import time
import warnings
from copy import copy
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.differentiation.derivatives import first_derivative
from optimagic.differentiation.numdiff_options import NumdiffOptions
from optimagic.exceptions import UserFunctionRuntimeError, get_traceback
from optimagic.logging.logger import LogStore
from optimagic.logging.types import IterationState
from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
    SpecificFunctionValue,
)
from optimagic.optimization.history import History, HistoryEntry
from optimagic.parameters.bounds import Bounds
from optimagic.parameters.conversion import Converter
from optimagic.typing import (
    AggregationLevel,
    BatchEvaluator,
    Direction,
    ErrorHandling,
    EvalTask,
    PyTree,
)


@dataclass(frozen=True)
class InternalBounds(Bounds):
    lower: NDArray[np.float64] | None
    upper: NDArray[np.float64] | None
    soft_lower: None = None
    soft_upper: None = None


class InternalOptimizationProblem:
    def __init__(
        self,
        fun: Callable[[PyTree], SpecificFunctionValue],
        jac: Callable[[PyTree], PyTree] | None,
        fun_and_jac: Callable[[PyTree], tuple[SpecificFunctionValue, PyTree]] | None,
        converter: Converter,
        solver_type: AggregationLevel,
        direction: Direction,
        bounds: InternalBounds,
        numdiff_options: NumdiffOptions,
        error_handling: ErrorHandling,
        error_penalty_func: Callable[
            [NDArray[np.float64]],
            tuple[SpecificFunctionValue, NDArray[np.float64]],
        ],
        batch_evaluator: BatchEvaluator,
        linear_constraints: list[dict[str, Any]] | None,
        nonlinear_constraints: list[dict[str, Any]] | None,
        logger: LogStore[Any, Any] | None,
        # TODO: add hess and hessp
    ):
        self._fun = fun
        self._jac = jac
        self._fun_and_jac = fun_and_jac
        self._converter = converter
        self._solver_type = solver_type
        self._direction = direction
        self._bounds = bounds
        self._numdiff_options = numdiff_options
        self._error_handling = error_handling
        self._error_penalty_func = error_penalty_func
        self._batch_evaluator = batch_evaluator
        self._history = History(direction)
        self._linear_constraints = linear_constraints
        self._nonlinear_constraints = nonlinear_constraints
        self._logger = logger
        self._step_id: int | None = None

    # ==================================================================================
    # Public methods used by optimizers
    # ==================================================================================

    def fun(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Evaluate the objective function at x.

        Args:
            x: The parameter vector at which to evaluate the objective function.

        Returns:
            The function value at x. This is a scalar for scalar problems and an array
                for least squares  or likelihood problems.

        """
        fun_value, hist_entry = self._evaluate_fun(x)
        self._history.add_entry(hist_entry)
        return fun_value

    def jac(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the first derivative at x.

        Args:
            x: The parameter vector at which to evaluate the first derivative.

        Returns:
            The first derivative at x. This is a 1d array for scalar problems (the
                gradient) and a 2d array for least squares or likelihood problems (the
                Jacobian).

        """
        jac_value, hist_entry = self._evaluate_jac(x)
        self._history.add_entry(hist_entry)
        return jac_value

    def fun_and_jac(
        self, x: NDArray[np.float64]
    ) -> tuple[float | NDArray[np.float64], NDArray[np.float64]]:
        """Simultaneously evaluate the objective function and its first derivative.

        See .fun and .jac for details.

        """
        fun_and_jac_value, hist_entry = self._evaluate_fun_and_jac(x)
        self._history.add_entry(hist_entry)
        return fun_and_jac_value

    def batch_fun(
        self,
        x_list: list[NDArray[np.float64]],
        n_cores: int,
        batch_size: int | None = None,
    ) -> list[float | NDArray[np.float64]]:
        """Parallelized batch version of .fun.

        Args:
            x_list: A list of parameter vectors at which to evaluate the objective
                function.
            n_cores: The number of cores to use for the parallel evaluation.
            batch_size: Batch size that can be used by some algorithms to simulate
                the behavior under parallelization on more cores than are actually
                available. Only used by `criterion_plots` and benchmark plots.

        Returns:
            A list of function values at the points in x_list. See .fun for details.

        """
        batch_size = n_cores if batch_size is None else batch_size
        batch_result = self._batch_evaluator(
            func=self._evaluate_fun,
            arguments=x_list,
            n_cores=n_cores,
            # This should always be raise because errors are already handled
            error_handling="raise",
        )
        fun_values = [result[0] for result in batch_result]
        hist_entries = [result[1] for result in batch_result]
        self._history.add_batch(hist_entries, batch_size)

        return fun_values

    def batch_jac(
        self,
        x_list: list[NDArray[np.float64]],
        n_cores: int,
        batch_size: int | None = None,
    ) -> list[NDArray[np.float64]]:
        """Parallelized batch version of .jac.

        Args:
            x_list: A list of parameter vectors at which to evaluate the first
                derivative.
            n_cores: The number of cores to use for the parallel evaluation.
            batch_size: Batch size that can be used by some algorithms to simulate
                the behavior under parallelization on more cores than are actually
                available. Only used by `criterion_plots` and benchmark plots.

        Returns:
            A list of first derivatives at the points in x_list. See .jac for details.

        """
        batch_size = n_cores if batch_size is None else batch_size

        batch_result = self._batch_evaluator(
            func=self._evaluate_jac,
            arguments=x_list,
            n_cores=n_cores,
            # This should always be raise because errors are already handled
            error_handling="raise",
        )
        jac_values = [result[0] for result in batch_result]
        hist_entries = [result[1] for result in batch_result]
        self._history.add_batch(hist_entries, batch_size)
        return jac_values

    def batch_fun_and_jac(
        self,
        x_list: list[NDArray[np.float64]],
        n_cores: int,
        batch_size: int | None = None,
    ) -> list[tuple[float | NDArray[np.float64], NDArray[np.float64]]]:
        """Parallelized batch version of .fun_and_jac.

        Args:
            x_list: A list of parameter vectors at which to evaluate the objective
                function and its first derivative.
            n_cores: The number of cores to use for the parallel evaluation.
            batch_size: Batch size that can be used by some algorithms to simulate
                the behavior under parallelization on more cores than are actually
                available. Only used by `criterion_plots` and benchmark plots.

        Returns:
            A list of tuples containing the function value and the first derivative
                at the points in x_list. See .fun_and_jac for details.

        """
        batch_size = n_cores if batch_size is None else batch_size
        batch_result = self._batch_evaluator(
            func=self._evaluate_fun_and_jac,
            arguments=x_list,
            n_cores=n_cores,
            # This should always be raise because errors are already handled
            error_handling="raise",
        )
        fun_and_jac_values = [result[0] for result in batch_result]
        hist_entries = [result[1] for result in batch_result]
        self._history.add_batch(hist_entries, batch_size)

        return fun_and_jac_values

    def exploration_fun(
        self,
        x_list: list[NDArray[np.float64]],
        n_cores: int,
        batch_size: int | None = None,
    ) -> list[float]:
        batch_size = n_cores if batch_size is None else batch_size
        batch_result = self._batch_evaluator(
            func=self._evaluate_exploration_fun,
            arguments=x_list,
            n_cores=n_cores,
            # This should always be raise because errors are already handled
            error_handling="raise",
        )
        fun_values = [result[0] for result in batch_result]
        hist_entries = [result[1] for result in batch_result]
        self._history.add_batch(hist_entries, batch_size)

        return fun_values

    def with_new_history(self) -> Self:
        new = copy(self)
        new._history = History(self.direction)
        return new

    def with_error_handling(self, error_handling: ErrorHandling) -> Self:
        new = copy(self)
        new._error_handling = error_handling
        return new

    def with_step_id(self, step_id: int) -> Self:
        new = copy(self)
        new._step_id = step_id
        return new

    # ==================================================================================
    # Public attributes
    # ==================================================================================

    @property
    def bounds(self) -> InternalBounds:
        """Bounds of the optimization problem."""
        return self._bounds

    @property
    def converter(self) -> Converter:
        """Converter between external and internal parameter representation.

        The converter transforms parameters between their user-provided
        representation (the external representation) and the flat numpy array used
        by the optimizer (the internal representation).

        This transformation includes:
        - Flattening and unflattening of pytree structures.
        - Applying parameter constraints via reparametrizations.
        - Scaling and unscaling of parameter values.

        The Converter object provides the following main attributes:

        - ``params_to_internal``: Callable that converts a pytree of external
          parameters to a flat numpy array of internal parameters.
        - ``params_from_internal``: Callable that converts a flat numpy array of
          internal parameters to a pytree of external parameters.
        - ``derivative_to_internal``: Callable that converts the derivative
          from the external parameter space to the internal space.
        - ``has_transforming_constraints``: Boolean that is True if the conversion
          involves constraints that are handled by reparametrization.

        Examples:
            The converter is particularly useful for algorithms that require initial
            values in the internal (flat) parameter space, while allowing the user
            to specify these values in the more convenient external (pytree) format.

            Here's how an optimization algorithm might use the converter internally
            to prepare parameters for the optimizer:

                >>> from optimagic.optimization.internal_optimization_problem import (
                ...     SphereExampleInternalOptimizationProblem
                ... )
                >>> import numpy as np
                >>>
                >>> # Optimization problem instance.
                >>> problem = SphereExampleInternalOptimizationProblem()
                >>>
                >>> # User provided parameters in external format.
                >>> user_params = np.array([1.0, 2.0, 3.0])
                >>>
                >>> # Convert to internal format for optimization algorithms.
                >>> internal_params = problem.converter.params_to_internal(user_params)
                >>> internal_params
                array([1., 2., 3.])

        """
        return self._converter

    @property
    def linear_constraints(self) -> list[dict[str, Any]] | None:
        # TODO: write a docstring as soon as we actually use this
        return self._linear_constraints

    @property
    def nonlinear_constraints(self) -> list[dict[str, Any]] | None:
        """Internal representation of nonlinear constraints.

        Compared to the user provided constraints, we have done the following
        transformations:

        1. The constraint a <= g(x) <= b is transformed to h(x) >= 0, where h(x) is
        - h(x) = g(x), if a == 0 and b == inf
        - h(x) = g(x) - a, if a != 0 and b == inf
        - h(x) = (g(x) - a, -g(x) + b) >= 0, if a != 0 and b != inf.

        2. The equality constraint g(x) = v is transformed to h(x) >= 0, where
        h(x) = (g(x) - v, -g(x) + v).

        3. Vector constraints are transformed to a list of scalar constraints.
        g(x) = (g1(x), g2(x), ...) >= 0 is transformed to (g1(x) >= 0, g2(x) >= 0, ...).

        4. The constraint function (defined on a selection of user-facing parameters) is
        transformed to be evaluated on the internal parameters.

        """
        return self._nonlinear_constraints

    @property
    def direction(self) -> Direction:
        """Direction of the optimization problem."""
        return self._direction

    @property
    def history(self) -> History:
        """History container for the optimization problem."""
        return self._history

    @property
    def logger(self) -> LogStore[Any, Any] | None:
        """Logger for the optimization problem."""
        return self._logger

    # ==================================================================================
    # Implementation of the public functions; The main difference is that the lower-
    # level implementations return a history entry instead of adding it to the history
    # directly so they can be called in parallel!
    # ==================================================================================

    def _evaluate_fun(
        self, x: NDArray[np.float64]
    ) -> tuple[float | NDArray[np.float64], HistoryEntry]:
        fun_value, hist_entry, log_entry = self._pure_evaluate_fun(x)

        if self._logger:
            self._logger.iteration_store.insert(log_entry)

        return fun_value, hist_entry

    def _evaluate_jac(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], HistoryEntry]:
        if self._jac is not None:
            jac_value, hist_entry, log_entry = self._pure_evaluate_jac(x)
        else:
            if self._fun_and_jac is not None:
                (_, jac_value), hist_entry, log_entry = self._pure_evaluate_fun_and_jac(
                    x
                )
            else:
                (_, jac_value), hist_entry, log_entry = (
                    self._pure_evaluate_numerical_fun_and_jac(x)
                )

            hist_entry = replace(hist_entry, task=EvalTask.JAC)

        if self._logger:
            self._logger.iteration_store.insert(log_entry)

        return jac_value, hist_entry

    def _evaluate_exploration_fun(
        self, x: NDArray[np.float64]
    ) -> tuple[float, HistoryEntry]:
        fun_value, hist_entry, log_entry = self._pure_exploration_fun(x)

        if self._logger:
            self._logger.iteration_store.insert(log_entry)

        return fun_value, hist_entry

    def _evaluate_fun_and_jac(
        self, x: NDArray[np.float64]
    ) -> tuple[tuple[float | NDArray[np.float64], NDArray[np.float64]], HistoryEntry]:
        if self._fun_and_jac is not None:
            (fun_value, jac_value), hist_entry, log_entry = (
                self._pure_evaluate_fun_and_jac(x)
            )
        elif self._jac is not None:
            fun_value, hist_entry, log_entry_fun = self._pure_evaluate_fun(x)
            jac_value, _, log_entry_jac = self._pure_evaluate_jac(x)
            hist_entry = replace(hist_entry, task=EvalTask.FUN_AND_JAC)
            log_entry = log_entry_fun.combine(log_entry_jac)
        else:
            (fun_value, jac_value), hist_entry, log_entry = (
                self._pure_evaluate_numerical_fun_and_jac(x)
            )

        if self._logger:
            self._logger.iteration_store.insert(log_entry)

        return (fun_value, jac_value), hist_entry

    # ==================================================================================
    # Atomic evaluations of user provided functions or numerical derivatives
    # ==================================================================================

    def _pure_evaluate_fun(
        self, x: NDArray[np.float64]
    ) -> tuple[float | NDArray[np.float64], HistoryEntry, IterationState]:
        """Evaluate fun and handle exceptions.

        This function does all the conversions from x to params and from
        SpecificFunctionValue to the internal value, including a sign flip for
        maximization.

        If any exception occurs during the evaluation of fun and error handling is set
        to CONTINUE, the fun value is replaced by a penalty value and a warning is
        issued.

        """
        start_time = time.perf_counter()
        params = self._converter.params_from_internal(x)
        traceback: None | str = None
        try:
            fun_value = self._fun(params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if self._error_handling in (
                ErrorHandling.RAISE,
                ErrorHandling.RAISE_STRICT,
            ):
                msg = "An error occurred when evaluating fun during optimization."
                raise UserFunctionRuntimeError(msg) from e
            else:
                traceback = get_traceback()
                msg = (
                    "The following exception was caught when evaluating fun during "
                    "optimization. The fun value was replaced by a penalty value to "
                    f"continue with the optimization.:\n\n{traceback}"
                )
                warnings.warn(msg)
                fun_value, _ = self._error_penalty_func(x)

        algo_fun_value, hist_fun_value = _process_fun_value(
            value=fun_value, solver_type=self._solver_type, direction=self._direction
        )
        stop_time = time.perf_counter()

        hist_entry = HistoryEntry(
            params=params,
            fun=hist_fun_value,
            start_time=start_time,
            stop_time=stop_time,
            task=EvalTask.FUN,
        )

        log_entry = IterationState(
            params=params,
            timestamp=start_time,
            scalar_fun=hist_fun_value,
            valid=not bool(traceback),
            raw_fun=fun_value,
            step=self._step_id,
            exceptions=traceback,
        )

        return algo_fun_value, hist_entry, log_entry

    def _pure_evaluate_jac(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], HistoryEntry, IterationState]:
        if self._jac is None:
            raise ValueError("The jac function is not defined.")

        start_time = time.perf_counter()
        traceback: None | str = None

        params = self._converter.params_from_internal(x)
        try:
            jac_value = self._jac(params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if self._error_handling in (
                ErrorHandling.RAISE,
                ErrorHandling.RAISE_STRICT,
            ):
                msg = "An error occurred when evaluating jac during optimization."
                raise UserFunctionRuntimeError(msg) from e
            else:
                traceback = get_traceback()

                msg = (
                    "The following exception was caught when evaluating jac during "
                    "optimization. The jac value was replaced by a penalty value to "
                    f"continue with the optimization.:\n\n{traceback}"
                )
                warnings.warn(msg)
                _, jac_value = self._error_penalty_func(x)

        out_jac = _process_jac_value(
            value=jac_value, direction=self._direction, converter=self._converter, x=x
        )
        _assert_finite_jac(
            out_jac=out_jac, jac_value=jac_value, params=params, origin="jac"
        )

        stop_time = time.perf_counter()

        hist_entry = HistoryEntry(
            params=params,
            fun=None,
            start_time=start_time,
            stop_time=stop_time,
            task=EvalTask.JAC,
        )

        log_entry = IterationState(
            params=params,
            timestamp=start_time,
            scalar_fun=None,
            valid=not bool(traceback),
            raw_fun=None,
            step=self._step_id,
            exceptions=traceback,
        )

        return out_jac, hist_entry, log_entry

    def _pure_evaluate_numerical_fun_and_jac(
        self, x: NDArray[np.float64]
    ) -> tuple[
        tuple[float | NDArray[np.float64], NDArray[np.float64]],
        HistoryEntry,
        IterationState,
    ]:
        start_time = time.perf_counter()
        traceback: None | str = None

        def func(x: NDArray[np.float64]) -> SpecificFunctionValue:
            p = self._converter.params_from_internal(x)
            return self._fun(p)

        try:
            numdiff_res = first_derivative(
                func,
                x,
                bounds=self._bounds,
                **asdict(self._numdiff_options),
                unpacker=lambda x: x.internal_value(self._solver_type),
                error_handling="raise_strict",
            )
            fun_value = numdiff_res.func_value
            jac_value = numdiff_res.derivative
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if self._error_handling in (
                ErrorHandling.RAISE,
                ErrorHandling.RAISE_STRICT,
            ):
                msg = (
                    "An error occurred when evaluating a numerical derivative "
                    "during optimization."
                )
                raise UserFunctionRuntimeError(msg) from e
            else:
                traceback = get_traceback()

                msg = (
                    "The following exception was caught when calculating a "
                    "numerical derivative during optimization. The jac value was "
                    "replaced by a penalty value to continue with the optimization."
                    f":\n\n{traceback}"
                )
                warnings.warn(msg)
                fun_value, jac_value = self._error_penalty_func(x)

        _assert_finite_jac(
            out_jac=jac_value,
            jac_value=jac_value,
            params=self._converter.params_from_internal(x),
            origin="numerical",
        )

        algo_fun_value, hist_fun_value = _process_fun_value(
            value=fun_value,  # type: ignore
            solver_type=self._solver_type,
            direction=self._direction,
        )

        if self._direction == Direction.MAXIMIZE:
            jac_value = -jac_value

        stop_time = time.perf_counter()

        hist_entry = HistoryEntry(
            params=self._converter.params_from_internal(x),
            fun=hist_fun_value,
            start_time=start_time,
            stop_time=stop_time,
            task=EvalTask.FUN_AND_JAC,
        )

        log_entry = IterationState(
            params=self._converter.params_from_internal(x),
            timestamp=start_time,
            scalar_fun=hist_fun_value,
            valid=not bool(traceback),
            raw_fun=fun_value,
            step=self._step_id,
            exceptions=traceback,
        )

        return (algo_fun_value, jac_value), hist_entry, log_entry

    def _pure_exploration_fun(
        self, x: NDArray[np.float64]
    ) -> tuple[float, HistoryEntry, IterationState]:
        start_time = time.perf_counter()
        params = self._converter.params_from_internal(x)
        traceback: None | str = None

        try:
            fun_value = self._fun(params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            traceback = get_traceback()

            msg = (
                "The following exception was caught when evaluating fun during the "
                "exploration phase of a multistart optimization. The fun value was "
                "replaced by a penalty value to continue with the "
                f"optimization.:\n\n{traceback}"
            )
            warnings.warn(msg)
            fun_value, _ = self._error_penalty_func(x)

        if not traceback:
            algo_fun_value, hist_fun_value = _process_fun_value(
                value=fun_value,
                # For exploration we always need a scalar value
                solver_type=AggregationLevel.SCALAR,
                direction=self._direction,
            )
        else:
            algo_fun_value = -np.inf
            hist_fun_value = -np.inf
            if self._direction == Direction.MAXIMIZE:
                hist_fun_value = np.inf

        stop_time = time.perf_counter()

        hist_entry = HistoryEntry(
            params=params,
            fun=hist_fun_value,
            start_time=start_time,
            stop_time=stop_time,
            task=EvalTask.EXPLORATION,
        )

        log_entry = IterationState(
            params=params,
            timestamp=start_time,
            scalar_fun=hist_fun_value,
            valid=not bool(traceback),
            raw_fun=fun_value,
            step=self._step_id,
            exceptions=traceback,
        )

        return cast(float, algo_fun_value), hist_entry, log_entry

    def _pure_evaluate_fun_and_jac(
        self, x: NDArray[np.float64]
    ) -> tuple[
        tuple[float | NDArray[np.float64], NDArray[np.float64]],
        HistoryEntry,
        IterationState,
    ]:
        if self._fun_and_jac is None:
            raise ValueError("The fun_and_jac function is not defined.")

        start_time = time.perf_counter()
        traceback: None | str = None
        params = self._converter.params_from_internal(x)

        try:
            fun_value, jac_value = self._fun_and_jac(params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if self._error_handling in (
                ErrorHandling.RAISE,
                ErrorHandling.RAISE_STRICT,
            ):
                msg = (
                    "An error occurred when evaluating fun_and_jac during optimization."
                )
                raise UserFunctionRuntimeError(msg) from e
            else:
                traceback = get_traceback()
                msg = (
                    "The following exception was caught when evaluating fun_and_jac "
                    "during optimization. The fun and jac values were replaced by "
                    f"penalty values to continue with the optimization.:\n\n{traceback}"
                )
                warnings.warn(msg)

                fun_value, jac_value = self._error_penalty_func(x)

        algo_fun_value, hist_fun_value = _process_fun_value(
            value=fun_value, solver_type=self._solver_type, direction=self._direction
        )

        if traceback:
            out_jac = jac_value
        else:
            out_jac = self._converter.derivative_to_internal(jac_value, x)

        if self._direction == Direction.MAXIMIZE:
            out_jac = -out_jac

        _assert_finite_jac(
            out_jac=out_jac, jac_value=jac_value, params=params, origin="fun_and_jac"
        )

        stop_time = time.perf_counter()

        hist_entry = HistoryEntry(
            params=params,
            fun=hist_fun_value,
            start_time=start_time,
            stop_time=stop_time,
            task=EvalTask.FUN_AND_JAC,
        )

        log_entry = IterationState(
            params=params,
            timestamp=start_time,
            scalar_fun=hist_fun_value,
            valid=not bool(traceback),
            raw_fun=fun_value,
            step=self._step_id,
            exceptions=traceback,
        )

        return (algo_fun_value, out_jac), hist_entry, log_entry


def _assert_finite_jac(
    out_jac: NDArray[np.float64],
    jac_value: PyTree,
    params: PyTree,
    origin: Literal["numerical", "jac", "fun_and_jac"],
) -> None:
    """Check for infinite and NaN values in the Jacobian and raise an error if found.

    Args:
        out_jac: internal processed Jacobian to check for finiteness.
        jac_value: original Jacobian value as returned by the user function,
        params: user-facing parameter representation at evaluation point.
        origin: Source of Jacobian calculation, for the error message.

    Raises:
        UserFunctionRuntimeError:
            If any infinite or NaN values are found in the Jacobian.

    """
    if not np.all(np.isfinite(out_jac)):
        if origin == "jac" or "fun_and_jac":
            msg = (
                "The optimization failed because the derivative provided via "
                f"{origin} contains infinite or NaN values."
                "\nPlease validate the derivative function."
            )
        elif origin == "numerical":
            msg = (
                "The optimization failed because the numerical derivative "
                "(computed using fun) contains infinite or NaN values."
                "\nPlease validate the criterion function or try a different optimizer."
            )
        msg += (
            f"\nParameters at evaluation point: {params}\nJacobian values: {jac_value}"
        )
        raise UserFunctionRuntimeError(msg)


def _process_fun_value(
    value: SpecificFunctionValue,
    solver_type: AggregationLevel,
    direction: Direction,
) -> tuple[float | NDArray[np.float64], float]:
    """Post-process a function value for use by the algorithm and as history entry.

    The sign flip for maximization is only applied to the value that will be passed to
    the algorithm.

    Args:
        value: The function value.
        solver_type: The aggregation level of the solver.
        direction: The direction of optimization.

    Returns:
        A tuple of the function value for the algorithm and the function value for the
        history entry.

    """
    algo_value = value.internal_value(solver_type)
    history_value = cast(float, value.internal_value(AggregationLevel.SCALAR))
    if direction == Direction.MAXIMIZE:
        algo_value = -algo_value

    return algo_value, history_value


def _process_jac_value(
    value: SpecificFunctionValue,
    direction: Direction,
    converter: Converter,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Post-process a for use by the algorithm.

    Args:
        value: The Jacobian value.
        direction: The direction of optimization.
        converter: The converter object.

    Returns:
        The Jacobian value for the algorithm.

    """

    out_value = converter.derivative_to_internal(value, x)
    if direction == Direction.MAXIMIZE:
        out_value = -out_value

    return out_value


class SphereExampleInternalOptimizationProblem(InternalOptimizationProblem):
    """Super simple example of an internal optimization problem.

    This can be used to test algorithm wrappers or to familiarize yourself with the
    internal optimization problem interface.

    Args:

    """

    def __init__(
        self,
        solver_type: AggregationLevel = AggregationLevel.SCALAR,
        binding_bounds: bool = False,
    ) -> None:
        _fun_dict = {
            AggregationLevel.SCALAR: lambda x: ScalarFunctionValue(x @ x),
            AggregationLevel.LIKELIHOOD: lambda x: LikelihoodFunctionValue(x**2),
            AggregationLevel.LEAST_SQUARES: lambda x: LeastSquaresFunctionValue(x),
        }

        _jac_dict = {
            AggregationLevel.SCALAR: lambda x: 2 * x,
            AggregationLevel.LIKELIHOOD: lambda x: 2 * x,
            AggregationLevel.LEAST_SQUARES: lambda x: np.eye(len(x)),
        }

        fun = _fun_dict[solver_type]
        jac = _jac_dict[solver_type]
        fun_and_jac = lambda x: (fun(x), jac(x))

        converter = Converter(
            params_to_internal=lambda x: x,
            params_from_internal=lambda x: x,
            derivative_to_internal=lambda x, x0: x,
            has_transforming_constraints=False,
        )

        direction = Direction.MINIMIZE

        if binding_bounds:
            lb = np.arange(10, dtype=np.float64) - 7.0
            ub = np.arange(10, dtype=np.float64) - 3.0
            self._x_opt = np.array([-3, -2, -1, 0, 0, 0, 0, 0, 1, 2.0])
        else:
            lb = np.full(10, -10, dtype=np.float64)
            ub = np.full(10, 10, dtype=np.float64)
            self._x_opt = np.zeros(10)

        bounds = InternalBounds(lb, ub)

        numdiff_options = NumdiffOptions()

        error_handling = ErrorHandling.RAISE

        error_penalty_func = fun_and_jac

        batch_evaluator = process_batch_evaluator("joblib")

        linear_constraints = None
        nonlinear_constraints = None

        logger = None

        super().__init__(
            fun=fun,
            jac=jac,
            fun_and_jac=fun_and_jac,
            converter=converter,
            solver_type=solver_type,
            direction=direction,
            bounds=bounds,
            numdiff_options=numdiff_options,
            error_handling=error_handling,
            error_penalty_func=error_penalty_func,
            batch_evaluator=batch_evaluator,
            linear_constraints=linear_constraints,
            nonlinear_constraints=nonlinear_constraints,
            logger=logger,
        )


class SphereExampleInternalOptimizationProblemWithConverter(
    InternalOptimizationProblem
):
    """Super simple example of an internal optimization problem with PyTree Converter.
    Note: params should be a dict with key-value pairs `"x{i}" : val .
    eg. `{'x0': 1, 'x1': 2, ...}`.

    The converter.params_to_internal method converts tree like
    `{'x0': 1, 'x1': 2, 'x2': 3 ...}` to flat array `[1,2,3 ...]` .

    The converter.params_from_internal method converts flat array `[1,2,3 ...]`
    to tree like `{'x0': 1, 'x1': 2, 'x2': 3 ...}`.

    The converter.derivative_to_internal converts derivative trees
    {'x0': 2,'x1': 4, } to flat arrays [2,4] and jacobian tree
    `{  "x0": {"x0": 1, "x1": 0, },
        "x1": {"x0": 0, "x1": 1, }`
    to NDArray [[1, 0,], [0, 1, ],]. }.
    This can be used to test algorithm wrappers or to familiarize yourself
    with the internal optimization problem interface.

    Args:

    """

    def __init__(
        self,
        solver_type: AggregationLevel = AggregationLevel.SCALAR,
        binding_bounds: bool = False,
    ) -> None:
        def sphere(params: PyTree) -> SpecificFunctionValue:
            out = sum([params[f"x{i}"] ** 2 for i in range(len(params))])
            return ScalarFunctionValue(out)

        def ls_sphere(params: PyTree) -> SpecificFunctionValue:
            out = [params[f"x{i}"] for i in range(len(params))]
            return LeastSquaresFunctionValue(out)

        def likelihood_sphere(params: PyTree) -> SpecificFunctionValue:
            out = [params[f"x{i}"] ** 2 for i in range(len(params))]
            return LikelihoodFunctionValue(out)

        _fun_dict = {
            AggregationLevel.SCALAR: sphere,
            AggregationLevel.LIKELIHOOD: likelihood_sphere,
            AggregationLevel.LEAST_SQUARES: ls_sphere,
        }

        def sphere_gradient(params: PyTree) -> PyTree:
            return {params[f"x{i}"]: 2 * v for i, v in enumerate(params.values())}

        def likelihood_sphere_gradient(params: PyTree) -> PyTree:
            return {params[f"x{i}"]: 2 * v for i, v in enumerate(params.values())}

        def ls_sphere_jac(params: PyTree) -> PyTree:
            return {
                f"x{i}": {f"x{j}": 1 if i == j else 0 for j in range(len(params))}
                for i in range(len(params))
            }

        _jac_dict = {
            AggregationLevel.SCALAR: sphere_gradient,
            AggregationLevel.LIKELIHOOD: likelihood_sphere_gradient,
            AggregationLevel.LEAST_SQUARES: ls_sphere_jac,
        }

        fun = _fun_dict[solver_type]
        jac = _jac_dict[solver_type]
        fun_and_jac = lambda x: (fun(x), jac(x))

        def params_flatten(params: PyTree) -> NDArray[np.float64]:
            return np.array([v for v in params.values()]).astype(float)

        def params_unflatten(x: NDArray[np.float64]) -> PyTree:
            return {f"x{i}": v for i, v in enumerate(x)}

        def derivative_flatten(tree: PyTree, x: NDArray[np.float64]) -> Any:
            if solver_type == AggregationLevel.LEAST_SQUARES:
                out = [list(row.values()) for row in tree.values()]
                return np.array(out)
            else:
                return params_flatten(tree)

        converter = Converter(
            params_to_internal=params_flatten,
            params_from_internal=params_unflatten,
            derivative_to_internal=derivative_flatten,
            has_transforming_constraints=False,
        )

        direction = Direction.MINIMIZE

        if binding_bounds:
            lb = np.arange(10, dtype=np.float64) - 7.0
            ub = np.arange(10, dtype=np.float64) - 3.0
            self._x_opt = np.array([-3, -2, -1, 0, 0, 0, 0, 0, 1, 2.0])
        else:
            lb = np.full(10, -10, dtype=np.float64)
            ub = np.full(10, 10, dtype=np.float64)
            self._x_opt = np.zeros(10)

        bounds = InternalBounds(lb, ub)

        numdiff_options = NumdiffOptions()

        error_handling = ErrorHandling.RAISE

        error_penalty_func = fun_and_jac

        batch_evaluator = process_batch_evaluator("joblib")

        linear_constraints = None
        nonlinear_constraints = None

        logger = None

        super().__init__(
            fun=fun,
            jac=jac,
            fun_and_jac=fun_and_jac,
            converter=converter,
            solver_type=solver_type,
            direction=direction,
            bounds=bounds,
            numdiff_options=numdiff_options,
            error_handling=error_handling,
            error_penalty_func=error_penalty_func,
            batch_evaluator=batch_evaluator,
            linear_constraints=linear_constraints,
            nonlinear_constraints=nonlinear_constraints,
            logger=logger,
        )
