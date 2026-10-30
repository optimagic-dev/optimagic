import numpy as np
import pytest
from numpy.testing import assert_allclose

from optimagic.exceptions import InvalidAlgoOptionError, UserFunctionRuntimeError
from optimagic.optimizers._bhhh.line_search import (
    LineSearch,
    line_search,
    next_step_length,
)
from optimagic.utilities import get_rng

INF = np.array([np.inf])


def _quadratic(best_step_length):
    """Contributions whose sum along direction -1 from x = 0 is minimal at t = best."""

    def fun(x):
        return np.array([(x[0] + best_step_length) ** 2, 0.0])

    return fun


def _search(
    fun,
    options=None,
    best_step_length=1.0,
    max_step_length=np.inf,
    expand=True,
    max_backtracks=20,
    max_fun_evals=1000,
):
    """Search from x = 0 along direction -1."""
    x = np.zeros(1)
    return line_search(
        fun=fun,
        x=x,
        criterion=float(np.sum(fun(x))),
        direction=-np.ones(1),
        # the derivative of (x + a)² at 0 is 2a, along direction -1 it is -2a
        slope=-2 * best_step_length,
        lower=-INF,
        upper=INF,
        options=LineSearch() if options is None else options,
        initial_step_length=1.0,
        max_step_length=max_step_length,
        expand=expand,
        max_backtracks=max_backtracks,
        max_fun_evals=max_fun_evals,
    )


def test_accepts_unit_step_if_it_is_optimal():
    res = _search(_quadratic(1.0), best_step_length=1.0)
    assert res.status == "success"
    assert res.step_length == 1
    assert res.n_backtracks == 0
    assert res.n_expansions == 0
    # one evaluation at t = 1 and one rejected doubling
    assert res.n_fun_evals == 2


def test_doubling_until_no_improvement():
    res = _search(_quadratic(10.0), best_step_length=10.0)
    assert res.step_length == 8
    assert res.n_expansions == 3
    assert_allclose(res.x, [-8.0])
    assert_allclose(res.criterion, 4.0)
    assert_allclose(res.fun_value, [4.0, 0.0])


def test_doubling_respects_max_step_length():
    res = _search(_quadratic(10.0), best_step_length=10.0, max_step_length=5.0)
    assert res.step_length == 4


def test_no_doubling_with_expansion_none():
    res = _search(
        _quadratic(10.0), options=LineSearch(expansion="none"), best_step_length=10.0
    )
    assert res.step_length == 1


def test_no_doubling_if_expand_is_false():
    res = _search(_quadratic(10.0), best_step_length=10.0, expand=False)
    assert res.step_length == 1


def test_backtracking_satisfies_armijo_condition():
    # the unit step overshoots by a factor of 10
    res = _search(_quadratic(0.1), best_step_length=0.1)
    assert res.status == "success"
    assert res.n_backtracks >= 1
    assert res.n_expansions == 0
    armijo_bound = 0.01 + 1e-4 * res.step_length * (-0.2)
    assert res.criterion <= armijo_bound


def test_non_finite_trial_points_are_rejected():
    def fun(x):
        out = (x + 1) ** 2
        return np.where(x < -0.05, np.nan, out)

    res = _search(fun, best_step_length=1.0)
    assert res.status == "success"
    assert res.step_length <= 0.05
    assert np.isfinite(res.criterion)


def test_infinite_trial_points_are_rejected():
    def fun(x):
        return np.where(x < -0.5, np.inf, (x + 1) ** 2)

    res = _search(fun, best_step_length=1.0)
    assert res.status == "success"
    assert res.step_length <= 0.5


def test_trial_points_that_raise_are_rejected():
    def fun(x):
        if x[0] < -0.5:
            raise UserFunctionRuntimeError("Evaluation failed.")
        return (x + 1) ** 2

    res = _search(fun, best_step_length=1.0)
    assert res.status == "success"
    assert res.step_length <= 0.5
    assert res.n_failed_fun_evals == 1
    assert isinstance(res.last_error, UserFunctionRuntimeError)


def test_failure_after_max_backtracks():
    def fun(x):
        return np.array([1.0 + abs(x[0])])

    res = _search(fun, best_step_length=1.0, max_backtracks=5)
    assert res.status == "max_backtracks"
    assert res.n_backtracks == 5
    assert res.fun_value is None
    assert_allclose(res.x, [0.0])


def test_failure_if_budget_is_exhausted():
    def fun(x):
        return np.array([1.0 + abs(x[0])])

    res = _search(fun, best_step_length=1.0, max_fun_evals=3)
    assert res.status == "max_fun_evals"
    assert res.n_fun_evals == 3


def test_no_evaluation_without_budget():
    res = _search(_quadratic(1.0), max_fun_evals=0)
    assert res.status == "max_fun_evals"
    assert res.n_fun_evals == 0


def test_doubling_stops_when_budget_is_exhausted():
    res = _search(_quadratic(10.0), best_step_length=10.0, max_fun_evals=2)
    assert res.status == "success"
    assert res.step_length == 2


def test_quadratic_interpolation_is_exact_for_quadratics():
    # f(t) = 1 - 2t + 5t² is minimal at t = 0.2; the rejected trial is t = 1
    t_new = next_step_length(t=1.0, value=4.0, previous=None, criterion=1.0, slope=-2)
    assert_allclose(t_new, 0.2)


def test_cubic_interpolation_is_exact_for_cubics():
    def f(t):
        return 1 - t + 2 * t**2 + 4 * t**3

    # f'(t) = -1 + 4t + 12t² is zero at t = 1/6, which is within the safeguards
    t_new = next_step_length(
        t=0.5, value=f(0.5), previous=(1.0, f(1.0)), criterion=1.0, slope=-1.0
    )
    assert_allclose(t_new, 1 / 6)


def test_overflow_in_interpolation_falls_back_to_halving():
    t_new = next_step_length(
        t=1e-200, value=1e300, previous=(1e-100, 1e300), criterion=0.0, slope=-1.0
    )
    assert t_new == 0.5e-200


def test_non_finite_value_shrinks_step_by_factor_ten():
    assert next_step_length(1.0, np.nan, None, 1.0, -1.0) == 0.1


@pytest.mark.parametrize("seed", range(20))
def test_interpolated_step_lengths_are_safeguarded(seed):
    rng = get_rng(seed)
    t = rng.uniform(0.01, 1)
    previous = (t * rng.uniform(2, 10), rng.uniform(-1, 10)) if seed % 2 else None
    value = rng.uniform(-1, 10)
    t_new = next_step_length(t, value, previous, criterion=0.0, slope=-rng.uniform())
    assert 0.1 * t <= t_new <= 0.5 * t


@pytest.mark.parametrize(
    "kwargs",
    [
        {"armijo_constant": 0.5},
        {"armijo_constant": 0},
        {"max_backtracks": 0},
        {"expansion": "quadratic"},
        {"max_step_factor": 0.5},
    ],
)
def test_invalid_options_raise(kwargs):
    with pytest.raises(InvalidAlgoOptionError):
        LineSearch(**kwargs)
