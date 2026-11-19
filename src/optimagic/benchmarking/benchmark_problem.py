"""Representation of benchmark functions and benchmark problems.

There are two levels:

- A ``BenchmarkFunction`` is the core of a problem: the objective function, optionally
  its derivative, the numbers of parameters it supports and properties like
  differentiability. It does not have start parameters. Example: the Rosenbrock
  function.
- A ``BenchmarkProblem`` is one concrete test case: a ``BenchmarkFunction`` with start
  parameters and optionally bounds, a known solution, noise and scaling. Example: the
  Rosenbrock function with the start parameters ``[-1.2, 1]``. Several benchmark
  problems can share the same benchmark function.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from optimagic import mark
from optimagic.benchmarking.noise import BenchmarkNoise
from optimagic.benchmarking.scaling import BenchmarkScaler
from optimagic.parameters.bounds import Bounds
from optimagic.typing import AggregationLevel

ObjectiveFunction = Callable[[NDArray[np.float64]], float | NDArray[np.float64]]
DerivativeFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]
FuncT = TypeVar("FuncT")


# ======================================================================================
# Benchmark functions
# ======================================================================================


class BenchmarkFunction(ABC):
    """Base class for the functions that benchmark problems are built from.

    To define a new function, subclass one of ``ScalarBenchmarkFunction``,
    ``LeastSquaresBenchmarkFunction`` or ``LikelihoodBenchmarkFunction`` and implement
    ``fun``. Parameters of a family of functions (e.g. the number of residuals) are
    attributes of the subclass. Use ``CallableBenchmarkFunction`` to wrap an existing
    Python function instead.

    A benchmark function is not tied to one number of parameters. The number of
    parameters is fixed by the start parameters of a ``BenchmarkProblem``.

    """

    problem_type: AggregationLevel
    """Whether ``fun`` returns a scalar, least-squares residuals or likelihood
    contributions. Set by the base classes for the three problem types."""

    differentiable: bool | None = None
    """Whether the function is differentiable at every point where it is defined. None
    if this is not known."""

    convex: bool | None = None
    """Whether the function is convex. For least-squares and likelihood functions this
    refers to the sum of squared residuals or the sum of the contributions. None if this
    is not known."""

    min_n_params: int = 1
    """The smallest number of parameters the function supports."""

    max_n_params: int | None = None
    """The largest number of parameters the function supports. None if there is no upper
    limit."""

    @abstractmethod
    def fun(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Evaluate the function.

        Args:
            x: The parameters, shape (n_params,).

        Returns:
            A float for scalar functions; the residuals of shape (n_residuals,) for
            least-squares functions; the contributions of shape (n_contributions,) for
            likelihood functions.

        """

    def jac(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the first derivative of ``fun``.

        Subclasses override this method if a closed form derivative is available.

        Args:
            x: The parameters, shape (n_params,).

        Returns:
            The gradient of shape (n_params,) for scalar functions; the jacobian of
            shape (n_residuals, n_params) or (n_contributions, n_params) for
            least-squares and likelihood functions.

        """
        raise NotImplementedError(
            f"{type(self).__name__} does not have a closed form derivative."
        )

    @property
    def has_jac(self) -> bool:
        """Whether a closed form derivative is available."""
        return type(self).jac is not BenchmarkFunction.jac

    def supports_n_params(self, n_params: int) -> bool:
        """Check if the function can be evaluated with n_params parameters.

        The default implementation checks ``min_n_params`` and ``max_n_params``.
        Override it for rules that cannot be expressed by those two numbers, e.g. if
        only even numbers of parameters are supported.

        Args:
            n_params: The number of parameters.

        Returns:
            True if the function supports n_params parameters.

        """
        upper = self.max_n_params
        return n_params >= self.min_n_params and (upper is None or n_params <= upper)

    def bounds(self, n_params: int) -> Bounds | None:  # noqa: ARG002
        """Get the bounds outside of which the function cannot be evaluated.

        Subclasses override this method if the function is not defined everywhere, e.g.
        because it takes the logarithm of a parameter.

        Args:
            n_params: The number of parameters.

        Returns:
            Bounds with arrays of shape (n_params,) or None if the function is defined
            everywhere.

        """
        return None


class ScalarBenchmarkFunction(BenchmarkFunction):
    """Base class for benchmark functions that return a scalar."""

    problem_type = AggregationLevel.SCALAR

    @abstractmethod
    def fun(self, x: NDArray[np.float64]) -> float:
        """Evaluate the function.

        Args:
            x: The parameters, shape (n_params,).

        Returns:
            The function value.

        """


class LeastSquaresBenchmarkFunction(BenchmarkFunction):
    """Base class for benchmark functions that return least-squares residuals.

    The function that is minimized is the sum of the squared residuals.

    """

    problem_type = AggregationLevel.LEAST_SQUARES

    @abstractmethod
    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the residuals.

        Args:
            x: The parameters, shape (n_params,).

        Returns:
            The residuals, shape (n_residuals,).

        """


class LikelihoodBenchmarkFunction(BenchmarkFunction):
    """Base class for benchmark functions that return likelihood contributions.

    The function that is minimized is the sum of the contributions. Since benchmark
    problems are minimization problems, the contributions are usually negative
    log-likelihood contributions.

    """

    problem_type = AggregationLevel.LIKELIHOOD

    @abstractmethod
    def fun(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the contributions.

        Args:
            x: The parameters, shape (n_params,).

        Returns:
            The contributions, shape (n_contributions,).

        """


@dataclass(frozen=True)
class CallableBenchmarkFunction(BenchmarkFunction):
    """Benchmark function that wraps an existing Python function.

    This is the quickest way to turn your own objective function into a benchmark
    function. In contrast to ``minimize``, the problem type is not inferred from the
    ``mark`` decorators but has to be passed explicitly.

    """

    func: ObjectiveFunction
    """The function that is wrapped. Takes parameters of shape (n_params,) and returns
    what ``fun`` of a benchmark function with the same ``problem_type`` returns."""

    problem_type: AggregationLevel
    """Whether ``func`` returns a scalar, least-squares residuals or likelihood
    contributions."""

    jac_func: DerivativeFunction | None = None
    """The first derivative of ``func`` or None if there is no closed form
    derivative."""

    differentiable: bool | None = None
    """Whether the function is differentiable at every point where it is defined. None
    if this is not known."""

    convex: bool | None = None
    """Whether the function is convex. None if this is not known."""

    min_n_params: int = 1
    """The smallest number of parameters the function supports."""

    max_n_params: int | None = None
    """The largest number of parameters the function supports. None if there is no upper
    limit."""

    def __post_init__(self) -> None:
        if not isinstance(self.problem_type, AggregationLevel):
            raise TypeError(
                "problem_type must be an AggregationLevel, not "
                f"{type(self.problem_type)}."
            )

    def fun(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        return self.func(x)

    def jac(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.jac_func is None:
            return super().jac(x)
        return self.jac_func(x)

    @property
    def has_jac(self) -> bool:
        return self.jac_func is not None


def aggregate_function_value(
    value: float | NDArray[np.float64], problem_type: AggregationLevel
) -> float:
    """Convert the output of a benchmark function to the scalar that is minimized.

    Args:
        value: What ``fun`` of a benchmark function returned.
        problem_type: The problem type of the benchmark function.

    Returns:
        The value itself for scalar functions; the sum of squared residuals for
        least-squares functions; the sum of the contributions for likelihood functions.

    """
    if problem_type == AggregationLevel.SCALAR:
        return float(value)
    arr = np.asarray(value, dtype=np.float64)
    if problem_type == AggregationLevel.LEAST_SQUARES:
        return float(arr @ arr)
    return float(arr.sum())


# ======================================================================================
# Benchmark problems
# ======================================================================================


@dataclass(frozen=True, eq=False)
class BenchmarkSolution:
    """The known solution of a benchmark problem."""

    fun: float
    """The function value at the solution. For least-squares problems this is the sum
    of squared residuals, for likelihood problems the sum of the contributions."""

    params: NDArray[np.float64] | None = None
    """The parameters at the solution, shape (n_params,). None if they are not known or
    not unique."""

    exact: bool = False
    """True if ``fun`` is known to be the global minimum. False if it is only the best
    value known so far, e.g. the result of a long run of a reliable optimizer. In the
    second case an optimizer can find a better value during a benchmark."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "fun", float(self.fun))
        if self.params is not None:
            object.__setattr__(self, "params", _as_read_only_array(self.params))


@dataclass(frozen=True, eq=False)
class BenchmarkProblem:
    """A concrete test case for optimizers.

    All arrays (start parameters, bounds and the parameters of the solution) are in the
    space the optimizer works in. If the problem is scaled, this is the scaled space.
    Use ``with_scaling`` and ``with_noise`` to add scaling or noise to a problem.

    """

    function: BenchmarkFunction
    """The benchmark function that is minimized."""

    start_params: NDArray[np.float64]
    """The parameters at which optimizers start, shape (n_params,). They fix the number
    of parameters of the problem. The array is read-only; copy it if you need to modify
    it."""

    solution: BenchmarkSolution | None = None
    """The known solution of the problem or None if no solution is known. Without a
    solution, benchmark plots and reports use the best function value any of the
    compared optimizers found."""

    bounds: Bounds | None = None
    """The bounds that are passed to the optimizer, with arrays of shape (n_params,).
    If None, the bounds outside of which the function cannot be evaluated are used (see
    ``BenchmarkFunction.bounds``)."""

    noise: BenchmarkNoise | None = None
    """The noise that is added to each function evaluation or None for no noise."""

    scaler: BenchmarkScaler | None = None
    """The scaler that was used to rescale the parameters or None if the problem is not
    scaled."""

    seed: int | None = None
    """The seed for the noise. Each optimization of the problem starts with a new
    random number generator with this seed, such that all optimizers see the same noise.
    If None, the noise differs between optimizations."""

    info: dict[str, Any] = field(default_factory=dict)
    """Additional information about the problem. The entries depend on the benchmark
    set."""

    def __post_init__(self) -> None:
        if not isinstance(self.function, BenchmarkFunction):
            raise TypeError(
                "function must be an instance of BenchmarkFunction, not "
                f"{type(self.function)}."
            )

        start_params = _as_read_only_array(self.start_params)
        if start_params.ndim != 1:
            raise ValueError(
                "start_params must be a one-dimensional array, not an array with "
                f"{start_params.ndim} dimensions."
            )
        object.__setattr__(self, "start_params", start_params)

        n_params = len(start_params)
        if not self.function.supports_n_params(n_params):
            raise ValueError(
                f"{type(self.function).__name__} does not support {n_params} "
                "parameters."
            )

        solution = self.solution
        if solution is not None and solution.params is not None:
            if np.shape(solution.params) != (n_params,):
                raise ValueError(
                    f"The solution has params of shape {np.shape(solution.params)} "
                    f"but the start_params have shape {(n_params,)}."
                )

        function_bounds = self.function.bounds(n_params)
        if self.scaler is not None:
            function_bounds = _scale_bounds(function_bounds, self.scaler)

        if self.bounds is None:
            object.__setattr__(self, "bounds", function_bounds)
        else:
            _fail_if_bounds_are_invalid(
                bounds=self.bounds, function_bounds=function_bounds, n_params=n_params
            )

    @property
    def n_params(self) -> int:
        """The number of parameters of the problem."""
        return len(self.start_params)

    @property
    def fun(self) -> ObjectiveFunction:
        """The objective function as the optimizer sees it, i.e. scaled and with noise.

        The function is marked with the problem type and can be passed to ``minimize``
        together with ``start_params``. Each access returns a new function whose noise
        starts from ``seed``.

        """
        rng = None if self.noise is None else np.random.default_rng(self.seed)
        func = _ProblemFunction(
            function=self.function, scaler=self.scaler, noise=self.noise, rng=rng
        )
        return _mark_problem_type(func, self.function.problem_type)

    @property
    def noise_free_fun(self) -> ObjectiveFunction:
        """The objective function as the optimizer sees it but without noise."""
        func = _ProblemFunction(
            function=self.function, scaler=self.scaler, noise=None, rng=None
        )
        return _mark_problem_type(func, self.function.problem_type)

    @property
    def jac(self) -> list[DerivativeFunction] | None:
        """Closed form derivatives of ``fun`` that can be passed to ``minimize``.

        The list contains one function for each type of optimizer that can use the
        derivative, e.g. the gradient of the sum of squared residuals for scalar
        optimizers and the jacobian of the residuals for least-squares optimizers.

        None if the benchmark function has no closed form derivative or if the problem
        is noisy or scaled. Optimizers then fall back to numerical derivatives.

        """
        if not self.function.has_jac or self.noise is not None:
            return None
        if self.scaler is not None:
            return None
        levels = _DERIVATIVE_LEVELS[self.function.problem_type]
        return [
            _mark_problem_type(_ProblemDerivative(self.function, level), level)
            for level in levels
        ]

    @cached_property
    def start_fun(self) -> float:
        """The function value at the start parameters, without noise.

        For least-squares problems this is the sum of squared residuals, for likelihood
        problems the sum of the contributions.

        """
        return aggregate_function_value(
            self.noise_free_fun(self.start_params),
            problem_type=self.function.problem_type,
        )

    def with_noise(self, noise: BenchmarkNoise | None, seed: int | None = None) -> Self:
        """Create a copy of the problem with different noise.

        Args:
            noise: The noise that is added to each function evaluation or None to
                remove the noise.
            seed: The seed for the noise.

        Returns:
            The modified copy.

        """
        if noise is not None and not isinstance(noise, BenchmarkNoise):
            raise TypeError(
                f"noise must be an instance of BenchmarkNoise or None, not "
                f"{type(noise)}."
            )
        return replace(self, noise=noise, seed=seed)

    def with_scaling(self, scaler: BenchmarkScaler) -> Self:
        """Create a rescaled copy of the problem.

        The start parameters, the bounds and the parameters of the solution are
        converted to the scaled space.

        Args:
            scaler: The scaler that is applied to the parameters.

        Returns:
            The modified copy.

        Raises:
            ValueError: If the problem is already scaled.

        """
        if not isinstance(scaler, BenchmarkScaler):
            raise TypeError(
                f"scaler must be an instance of BenchmarkScaler, not {type(scaler)}."
            )
        if self.scaler is not None:
            raise ValueError("The problem is already scaled.")

        solution = self.solution
        if solution is not None and solution.params is not None:
            solution = replace(solution, params=scaler.scale(solution.params))

        return replace(
            self,
            start_params=scaler.scale(self.start_params),
            solution=solution,
            bounds=_scale_bounds(self.bounds, scaler),
            scaler=scaler,
        )


class _ProblemFunction:
    """Objective function of a benchmark problem.

    This is a class and not a closure such that it can be pickled, which is required for
    running benchmarks in parallel.

    """

    def __init__(
        self,
        function: BenchmarkFunction,
        scaler: BenchmarkScaler | None,
        noise: BenchmarkNoise | None,
        rng: np.random.Generator | None,
    ) -> None:
        self._function = function
        self._scaler = scaler
        self._noise = noise
        self._rng = rng

    def __call__(self, params: NDArray[np.float64]) -> float | NDArray[np.float64]:
        x = params if self._scaler is None else self._scaler.unscale(params)
        fval = self._function.fun(x)
        if self._noise is None or self._rng is None:
            return fval
        if isinstance(fval, np.ndarray):
            array = cast(NDArray[np.float64], fval)
            return array + self._noise.draw_noise(array, params=x, rng=self._rng)
        value = float(fval)
        return value + self._noise.draw_noise(value, params=x, rng=self._rng)


_DERIVATIVE_LEVELS = {
    AggregationLevel.SCALAR: [AggregationLevel.SCALAR],
    AggregationLevel.LEAST_SQUARES: [
        AggregationLevel.SCALAR,
        AggregationLevel.LIKELIHOOD,
        AggregationLevel.LEAST_SQUARES,
    ],
    AggregationLevel.LIKELIHOOD: [
        AggregationLevel.SCALAR,
        AggregationLevel.LIKELIHOOD,
    ],
}
"""The types of optimizers for which a derivative can be calculated from the derivative
of a benchmark function with a given problem type."""


class _ProblemDerivative:
    """Derivative of a benchmark function for one type of optimizer.

    This is a class and not a closure such that it can be pickled.

    """

    def __init__(self, function: BenchmarkFunction, level: AggregationLevel) -> None:
        self._function = function
        self._level = level

    def __call__(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        jac = self._function.jac(params)
        problem_type = self._function.problem_type
        if problem_type == self._level:
            return jac
        if problem_type == AggregationLevel.LIKELIHOOD:
            # gradient of the sum of the contributions
            return jac.sum(axis=0)
        residuals = np.asarray(self._function.fun(params), dtype=np.float64)
        if self._level == AggregationLevel.SCALAR:
            # gradient of the sum of squared residuals
            return 2 * jac.T @ residuals
        # jacobian of the squared residuals
        return 2 * residuals.reshape(-1, 1) * jac


def _mark_problem_type(func: FuncT, problem_type: AggregationLevel) -> FuncT:
    markers: dict[AggregationLevel, Callable[[Any], Any]] = {
        AggregationLevel.SCALAR: mark.scalar,
        AggregationLevel.LEAST_SQUARES: mark.least_squares,
        AggregationLevel.LIKELIHOOD: mark.likelihood,
    }
    return markers[problem_type](func)


def _as_read_only_array(values: Any) -> NDArray[np.float64]:
    """Copy values into a float array that cannot be modified.

    Benchmark problems are shared between all users of a problem set. Read-only arrays
    make sure that nobody changes them by accident.

    """
    out = np.array(values, dtype=np.float64)
    out.setflags(write=False)
    return out


def _scale_bounds(bounds: Bounds | None, scaler: BenchmarkScaler) -> Bounds | None:
    if bounds is None:
        return None

    def _scale(arr: Any) -> NDArray[np.float64] | None:
        if arr is None:
            return None
        return scaler.scale(np.asarray(arr, dtype=np.float64))

    return Bounds(
        lower=_scale(bounds.lower),
        upper=_scale(bounds.upper),
        soft_lower=_scale(bounds.soft_lower),
        soft_upper=_scale(bounds.soft_upper),
    )


def _fail_if_bounds_are_invalid(
    bounds: Bounds, function_bounds: Bounds | None, n_params: int
) -> None:
    if not isinstance(bounds, Bounds):
        raise TypeError(
            f"bounds must be an instance of optimagic.Bounds, not {type(bounds)}."
        )

    lower, upper = _get_bound_arrays(bounds, n_params)
    function_lower, function_upper = _get_bound_arrays(function_bounds, n_params)
    if (lower < function_lower).any() or (upper > function_upper).any():
        raise ValueError(
            "The bounds of a benchmark problem must not be wider than the bounds "
            "outside of which its function cannot be evaluated."
        )


def _get_bound_arrays(
    bounds: Bounds | None, n_params: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert bounds to arrays of shape (n_params,), filling missing bounds."""
    lower = None if bounds is None else bounds.lower
    upper = None if bounds is None else bounds.upper
    out = []
    for bound, fill in [(lower, -np.inf), (upper, np.inf)]:
        if bound is None:
            out.append(np.full(n_params, fill))
        else:
            arr = np.asarray(bound, dtype=np.float64)
            if arr.shape != (n_params,):
                raise ValueError(
                    f"Bounds must have shape {(n_params,)}, not {arr.shape}."
                )
            out.append(arr)
    return out[0], out[1]
