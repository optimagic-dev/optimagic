"""Tests for SciPy-style callback(xk) support."""

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal as aae

import optimagic as om
from optimagic import mark
from optimagic.exceptions import InvalidFunctionError, InvalidKwargsError
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.optimize import minimize
from optimagic.typing import AggregationLevel


def test_callback_xk_is_called():
    """SciPy-style callback(xk) is invoked on objective evaluations."""
    xs = []

    def callback(xk):
        xs.append(np.asarray(xk))

    res = om.minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3, dtype=float),
        algorithm="scipy_neldermead",
        callback=callback,
    )

    assert len(xs) >= 1
    assert xs[0].shape == (3,)
    aaae(res.x, np.zeros(3), decimal=5)


def test_callback_matches_history_params():
    """Callback parameters match the collected optimization history."""
    xs = []

    def callback(xk):
        xs.append(np.asarray(xk))

    res = om.minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3, dtype=float),
        algorithm="scipy_neldermead",
        callback=callback,
    )

    assert res.history is not None
    assert len(xs) == len(res.history.params)
    for got, expected in zip(xs, res.history.params, strict=True):
        aaae(got, expected)


def test_callback_receives_external_params():
    """Callback gets external PyTree params, not the internal flat vector."""
    received = []

    def callback(xk):
        received.append(xk)

    params = {"a": np.array([1.0, 2.0]), "b": np.array([3.0])}

    def fun(p):
        return p["a"] @ p["a"] + p["b"] @ p["b"]

    res = om.minimize(
        fun=fun,
        params=params,
        algorithm="scipy_neldermead",
        callback=callback,
    )

    assert len(received) >= 1
    assert isinstance(received[0], dict)
    assert set(received[0]) == {"a", "b"}
    assert received[0]["a"].shape == (2,)
    assert received[0]["b"].shape == (1,)

    assert res.history is not None
    assert len(received) == len(res.history.params)
    for got, expected in zip(received, res.history.params, strict=True):
        assert set(got) == set(expected)
        aaae(got["a"], expected["a"])
        aaae(got["b"], expected["b"])


def test_callback_not_called_on_jac():
    """Callback runs next to history on fun, not on jac-only evaluations."""
    from optimagic.optimization.internal_optimization_problem import (
        SphereExampleInternalOptimizationProblem,
    )

    problem = SphereExampleInternalOptimizationProblem()
    calls = []
    problem._callback = lambda p: calls.append(np.asarray(p))

    x = np.ones(10)
    problem.jac(x)
    assert calls == []

    problem.fun(x)
    assert len(calls) == 1
    aaae(calls[0], x)


def test_invalid_callback_too_few_arguments():
    msg = "callback must have at least one free argument"

    def bad_callback():
        return None

    with pytest.raises(InvalidFunctionError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3, dtype=float),
            algorithm="scipy_neldermead",
            callback=bad_callback,
        )


def test_invalid_callback_too_many_required_arguments():
    msg = "Too few keyword arguments for callback"

    def bad_callback(xk, extra):
        return None

    with pytest.raises(InvalidKwargsError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3, dtype=float),
            algorithm="scipy_neldermead",
            callback=bad_callback,
        )


def test_invalid_callback_not_callable():
    with pytest.raises(InvalidFunctionError, match="callback must be a callable"):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3, dtype=float),
            algorithm="scipy_neldermead",
            callback="not-a-callable",
        )


@mark.minimizer(
    name="dummy_callback_parallel",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=False,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class _DummyParallelOptimizer(Algorithm):
    n_cores: int = 1
    batch_size: int = 1

    def _solve_internal_problem(self, problem, x0):
        xs = np.arange(15).repeat(len(x0)).reshape(15, len(x0))

        for iteration in range(3):
            start_index = iteration * 5
            problem.batch_fun(
                list(xs[start_index : start_index + 4]),
                n_cores=self.n_cores,
                batch_size=self.batch_size,
            )
            problem.fun(xs[start_index + 4])

        return InternalOptimizeResult(
            x=xs[-1],
            fun=5,
            success=True,
            n_fun_evals=15,
            n_iterations=3,
        )


def test_callback_history_with_parallel_optimizer():
    """History collected via callback matches optimagic history under parallelism."""
    collected = []

    def callback(xk):
        collected.append(np.asarray(xk))

    res = minimize(
        fun=lambda x: 5.0,
        params=np.arange(5, dtype=float),
        algorithm=_DummyParallelOptimizer,
        algo_options={"n_cores": 2, "batch_size": 2},
        callback=callback,
    )

    assert res.history is not None
    assert len(collected) == len(res.history.params)
    aae(collected, res.history.params)
