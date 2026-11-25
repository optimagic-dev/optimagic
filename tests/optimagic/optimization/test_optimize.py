"""Tests for (almost) algorithm independent properties of maximize and minimize."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om
from optimagic import mark
from optimagic.examples.criterion_functions import sos_scalar
from optimagic.exceptions import (
    InvalidFunctionError,
    InvalidNumdiffOptionsError,
    UserFunctionRuntimeError,
)
from optimagic.optimization.optimize import (
    MAX_PATHS_IN_ERROR_MESSAGE,
    maximize,
    minimize,
)


def test_sign_is_switched_back_after_maximization():
    params = pd.DataFrame()
    params["value"] = [1, 2, 3]
    res = maximize(
        lambda params: 1 - params["value"] @ params["value"],
        params=params,
        algorithm="scipy_lbfgsb",
    )

    assert np.allclose(res.fun, 1)


def test_scipy_lbfgsb_actually_calls_criterion_and_derivative():
    params = pd.DataFrame(data=np.ones((10, 1)), columns=["value"])

    def raising_crit_and_deriv(params):  # noqa: ARG001
        raise NotImplementedError("This should not be called.")

    with pytest.raises(InvalidFunctionError, match="Error while evaluating"):
        minimize(
            fun=sos_scalar,
            params=params,
            algorithm="scipy_lbfgsb",
            fun_and_jac=raising_crit_and_deriv,
        )


def test_with_invalid_numdiff_options():
    with pytest.raises(InvalidNumdiffOptionsError):
        minimize(
            fun=lambda x: x @ x,
            params=np.arange(5),
            algorithm="scipy_lbfgsb",
            numdiff_options={"bla": 15},  # ty:ignore[invalid-argument-type, invalid-key]
        )


# provided fun or fun_and_jac is provided
def test_with_optional_fun_argument():
    expected = np.zeros(5)
    res = minimize(
        fun_and_jac=lambda x: (x @ x, 2 * x),
        params=np.arange(5),
        algorithm="scipy_lbfgsb",
    )
    aaae(res.x, expected)


@pytest.mark.parametrize("container", [list, tuple])
def test_fun_and_jac_sequence_without_fun(container):
    with pytest.raises(NotImplementedError):
        minimize(
            fun_and_jac=container([lambda x: (x @ x, 2 * x)]),
            params=np.arange(5),
            algorithm="scipy_lbfgsb",
        )


# ======================================================================================
# The objective function has to be finite at the start params
# ======================================================================================


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_non_finite_scalar_fun_at_start_raises(value):
    def fun(x):
        return value if x[0] == 5 else x @ x

    with pytest.raises(InvalidFunctionError, match="not finite") as e:
        minimize(fun, params=np.array([5.0, 2.0]), algorithm="scipy_lbfgsb")
    # a scalar has no path
    assert "path" not in str(e.value)
    assert "multistart" in str(e.value)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_non_finite_scalar_fun_at_start_raises_with_maximize(value):
    def fun(x):
        return value if x[0] == 5 else -x @ x

    with pytest.raises(InvalidFunctionError, match="not finite"):
        maximize(fun, params=np.array([5.0, 2.0]), algorithm="scipy_lbfgsb")


def test_finite_fun_at_start_does_not_raise_with_maximize():
    res = maximize(lambda x: -x @ x, params=np.ones(2), algorithm="scipy_lbfgsb")
    aaae(res.params, np.zeros(2))


def test_non_finite_fun_at_start_raises_with_error_handling_continue():
    def fun(x):
        return np.nan if x[0] == 5 else x @ x

    with pytest.raises(InvalidFunctionError, match="not finite"):
        minimize(
            fun,
            params=np.array([5.0, 2.0]),
            algorithm="scipy_lbfgsb",
            error_handling="continue",
        )


@pytest.mark.parametrize("algorithm", ["scipy_lbfgsb", "scipy_neldermead"])
def test_non_finite_fun_at_start_raises_with_multistart(algorithm):
    def fun(x):
        return np.nan if x[0] == 5 else x @ x

    with pytest.raises(InvalidFunctionError, match="not finite"):
        minimize(
            fun,
            params=np.array([5.0, 2.0]),
            algorithm=algorithm,
            bounds=om.Bounds(lower=np.full(2, -10.0), upper=np.full(2, 10.0)),
            multistart=True,
        )


def test_non_finite_likelihood_contributions_at_start_raise_with_short_message():
    @mark.likelihood
    def fun(x):
        out = np.ones(20_000) * (x @ x)
        out[[1_500, 15_000]] = np.nan
        return out

    with pytest.raises(InvalidFunctionError, match="not finite") as e:
        minimize(fun, params=np.ones(2), algorithm="scipy_lbfgsb")
    msg = str(e.value)
    assert "2 of the 20000 entries" in msg
    assert "Their paths are: 1500, 15000" in msg
    # the output of the function is abbreviated
    assert "..." in msg
    assert len(msg) < 1_000


def test_non_finite_fun_at_start_raises_with_limited_number_of_paths():
    @mark.likelihood
    def fun(x):
        return np.full(1_000, np.nan)

    with pytest.raises(InvalidFunctionError, match="not finite") as e:
        minimize(fun, params=np.ones(2), algorithm="scipy_lbfgsb")
    msg = str(e.value)
    assert "1000 of the 1000 entries" in msg
    last_path = MAX_PATHS_IN_ERROR_MESSAGE - 1
    n_omitted = 1_000 - MAX_PATHS_IN_ERROR_MESSAGE
    assert f", {last_path} (and {n_omitted} more)" in msg


def test_non_finite_pytree_residuals_at_start_raise_with_paths():
    @mark.least_squares
    def fun(x):
        return {"a": x, "b": pd.Series([np.inf, 1.0], index=["c", "d"])}

    with pytest.raises(InvalidFunctionError, match="not finite") as e:
        minimize(fun, params=np.ones(2), algorithm="scipy_ls_lm")
    msg = str(e.value)
    assert "1 of the 4 entries" in msg
    assert "Its path is: b_c" in msg
    # small outputs are shown in full
    assert "'b':" in msg
    assert "c    inf" in msg


def test_overflow_in_aggregation_of_residuals_at_start_raises():
    @mark.least_squares
    def fun(x):
        return np.full(3, 1e200) * x[0]

    with pytest.raises(InvalidFunctionError, match="not finite") as e:
        minimize(fun, params=np.ones(2), algorithm="scipy_lbfgsb")
    assert "All 3 entries returned by the objective function are finite" in str(e.value)


# ======================================================================================
# User provided derivatives have to be finite at the start params if they are used
# ======================================================================================


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_non_finite_jac_at_start_raises(value):
    def jac(x):
        return np.array([2 * x[0], value, 2 * x[2]])

    with pytest.raises(
        InvalidFunctionError, match="provided via jac is not finite"
    ) as e:
        minimize(lambda x: x @ x, params=np.ones(3), algorithm="scipy_lbfgsb", jac=jac)
    msg = str(e.value)
    assert "1 of the 3 entries" in msg
    # for flat arrays, the paths are positions
    assert "Its path is: 1\n" in msg


def test_non_finite_jac_at_start_is_not_a_user_function_runtime_error():
    # Before, this was a UserFunctionRuntimeError that was raised by the algorithm's
    # first derivative evaluation. Code that catches this error does not catch the
    # error at the start params.
    def jac(x):
        return np.array([np.nan, 2 * x[1]])

    with pytest.raises(InvalidFunctionError) as e:
        minimize(lambda x: x @ x, params=np.ones(2), algorithm="scipy_lbfgsb", jac=jac)
    assert not isinstance(e.value, UserFunctionRuntimeError)


def test_non_finite_jac_after_start_is_a_user_function_runtime_error():
    def jac(x):
        return 2 * x if np.allclose(x, 1) else np.array([np.nan, 2 * x[1]])

    with pytest.raises(UserFunctionRuntimeError, match="infinite or NaN values"):
        minimize(lambda x: x @ x, params=np.ones(2), algorithm="scipy_lbfgsb", jac=jac)


def test_non_finite_jac_at_start_does_not_raise_for_derivative_free_algorithm():
    def jac(x):
        return np.array([np.inf, 2 * x[1]])

    res = minimize(
        lambda x: x @ x, params=np.ones(2), algorithm="scipy_neldermead", jac=jac
    )
    aaae(res.params, np.zeros(2), decimal=4)


def test_derivatives_are_not_evaluated_for_derivative_free_algorithm():
    def raise_error(x):
        raise ValueError("Derivatives should not be evaluated.")

    res = minimize(
        lambda x: x @ x,
        params=np.ones(2),
        algorithm="scipy_neldermead",
        jac=raise_error,
        fun_and_jac=raise_error,
    )
    aaae(res.params, np.zeros(2), decimal=4)


def test_non_finite_fun_and_jac_at_start_raises():
    def fun_and_jac(x):
        return x["a"] ** 2 + x["b"] ** 2, {"a": 2 * x["a"], "b": np.nan}

    with pytest.raises(InvalidFunctionError, match="fun_and_jac is not finite") as e:
        minimize(
            fun_and_jac=fun_and_jac,
            params={"a": 1.0, "b": 2.0},
            algorithm="scipy_lbfgsb",
        )
    msg = str(e.value)
    assert "Its path is: b\n" in msg
    assert "'b': nan" in msg


@pytest.mark.filterwarnings("ignore:overflow encountered")
def test_jac_that_is_only_non_finite_after_scaling_at_start_raises():
    def jac(x):
        return np.array([1e300, 2 * x[1]])

    with pytest.raises(InvalidFunctionError, match="jac is not finite") as e:
        minimize(
            lambda x: x @ x,
            params=np.array([1e10, 1.0]),
            algorithm="scipy_lbfgsb",
            jac=jac,
            scaling=True,
        )
    assert "All 2 entries of the derivative are finite" in str(e.value)


def test_non_finite_jac_of_single_param_at_start_raises_without_path():
    with pytest.raises(InvalidFunctionError, match="jac is not finite") as e:
        minimize(
            lambda x: x**2,
            params=1.0,
            algorithm="scipy_lbfgsb",
            jac=lambda x: np.nan,
        )
    assert "path" not in str(e.value)


# ======================================================================================
# fun_and_jac is not evaluated more often than necessary at the start params
# ======================================================================================


def _get_counting_fun_and_jac():
    calls = []

    def fun_and_jac(x):
        calls.append(x)
        return x @ x, 2 * x

    return fun_and_jac, calls


def test_fun_and_jac_without_fun_is_evaluated_once_at_start():
    fun_and_jac, calls = _get_counting_fun_and_jac()

    res = minimize(fun_and_jac=fun_and_jac, params=np.ones(2), algorithm="scipy_lbfgsb")

    aaae(res.params, np.zeros(2))
    # one evaluation at the start params and the evaluations done by the algorithm
    assert len(calls) - 1 == res.n_fun_evals


def test_fun_with_jac_equal_to_true_is_evaluated_once_at_start():
    fun_and_jac, calls = _get_counting_fun_and_jac()

    res = minimize(fun_and_jac, params=np.ones(2), algorithm="scipy_lbfgsb", jac=True)

    aaae(res.params, np.zeros(2))
    assert len(calls) - 1 == res.n_fun_evals


def test_fun_and_jac_is_evaluated_once_at_start_if_fun_is_provided():
    fun_and_jac, calls = _get_counting_fun_and_jac()

    res = minimize(
        lambda x: x @ x,
        fun_and_jac=fun_and_jac,
        params=np.ones(2),
        algorithm="scipy_lbfgsb",
    )

    aaae(res.params, np.zeros(2))
    assert len(calls) - 1 == res.n_fun_evals


def test_start_derivative_is_evaluated_with_fun_and_jac_kwargs():
    # The derivative at the start params is only finite with the fun_and_jac_kwargs.
    # It must not be taken from the first evaluation of fun, which gets fun_kwargs.
    def fun_and_jac(x, factor):
        return x @ x, 2 * x * factor

    res = minimize(
        fun_and_jac=fun_and_jac,
        fun_kwargs={"factor": np.nan},
        fun_and_jac_kwargs={"factor": 1.0},
        params=np.ones(2),
        algorithm="scipy_lbfgsb",
    )
    aaae(res.params, np.zeros(2))


# ======================================================================================
# A fun that is derived from fun_and_jac gets the same kwargs as fun_and_jac
# ======================================================================================


def _get_counting_fun_and_jac_with_target():
    calls = []

    def fun_and_jac(x, target):
        calls.append(x)
        diff = x - target
        return diff @ diff, 2 * diff

    return fun_and_jac, calls


@pytest.mark.parametrize("use_fun_kwargs", [True, False])
def test_fun_and_jac_without_fun_shares_kwargs(use_fun_kwargs):
    fun_and_jac, calls = _get_counting_fun_and_jac_with_target()
    target = np.array([1.0, 2.0])
    kwargs = {"target": target}

    res = minimize(
        fun_and_jac=fun_and_jac,
        params=np.zeros(2),
        algorithm="scipy_lbfgsb",
        fun_kwargs=kwargs if use_fun_kwargs else None,
        fun_and_jac_kwargs=None if use_fun_kwargs else kwargs,
    )

    aaae(res.params, target)
    assert len(calls) - 1 == res.n_fun_evals


@pytest.mark.parametrize("use_fun_kwargs", [True, False])
def test_fun_with_jac_equal_to_true_shares_kwargs(use_fun_kwargs):
    fun_and_jac, calls = _get_counting_fun_and_jac_with_target()
    target = np.array([1.0, 2.0])
    kwargs = {"target": target}

    res = minimize(
        fun_and_jac,
        params=np.zeros(2),
        algorithm="scipy_lbfgsb",
        jac=True,
        fun_kwargs=kwargs if use_fun_kwargs else None,
        fun_and_jac_kwargs=None if use_fun_kwargs else kwargs,
    )

    aaae(res.params, target)
    assert len(calls) - 1 == res.n_fun_evals


def test_fun_with_jac_equal_to_true_and_args():
    fun_and_jac, calls = _get_counting_fun_and_jac_with_target()
    target = np.array([1.0, 2.0])

    res = minimize(
        fun_and_jac,
        params=np.zeros(2),
        algorithm="scipy_lbfgsb",
        jac=True,
        args=(target,),
    )

    aaae(res.params, target)
    assert len(calls) - 1 == res.n_fun_evals


def test_start_derivative_of_unused_fun_and_jac_is_ignored():
    @mark.least_squares
    def fun_and_jac(x):
        return x, np.full((2, 2), np.nan)

    with pytest.warns(UserWarning, match="Falling back to numerical derivatives"):
        res = minimize(
            fun_and_jac=fun_and_jac, params=np.ones(2), algorithm="scipy_lbfgsb"
        )
    aaae(res.params, np.zeros(2))


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_marked_function_with_dict_output_removed_in_0_6_0_raises(direction):
    @mark.likelihood
    def fun(x):
        return {"value": float(x @ x), "contributions": x**2}

    with pytest.raises(InvalidFunctionError, match="0.6.0"):
        getattr(om, direction)(fun, params=np.ones(2), algorithm="scipy_lbfgsb")


def test_least_squares_function_with_dict_output_removed_in_0_6_0_raises():
    @mark.least_squares
    def fun(x):
        return {"value": float(x @ x), "root_contributions": x}

    with pytest.raises(InvalidFunctionError, match="0.6.0"):
        om.minimize(fun, params=np.ones(2), algorithm="scipy_ls_lm")
