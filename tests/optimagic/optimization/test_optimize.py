"""Tests for (almost) algorithm independent properties of maximize and minimize."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic import mark
from optimagic.examples.criterion_functions import sos_scalar
from optimagic.exceptions import InvalidFunctionError, InvalidNumdiffOptionsError
from optimagic.optimization.optimize import maximize, minimize


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


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_non_finite_scalar_fun_at_start_raises(value):
    def fun(x):
        return value if x[0] == 5 else x @ x

    with pytest.raises(InvalidFunctionError, match="not finite"):
        minimize(fun, params=np.array([5.0, 2.0]), algorithm="scipy_lbfgsb")


def test_non_finite_likelihood_contributions_at_start_raise_with_full_output():
    @mark.likelihood
    def fun(x):
        out = np.ones(2_000) * (x @ x)
        out[1_500] = np.nan
        return out

    with pytest.raises(InvalidFunctionError, match="not finite") as e:
        minimize(fun, params=np.ones(2), algorithm="scipy_lbfgsb")
    assert "nan" in str(e.value)
    assert "..." not in str(e.value)


def test_non_finite_pytree_residuals_at_start_raise_with_full_output():
    @mark.least_squares
    def fun(x):
        return {"a": x, "b": pd.Series([np.inf, 1.0], index=["c", "d"])}

    with pytest.raises(InvalidFunctionError, match="not finite") as e:
        minimize(fun, params=np.ones(2), algorithm="scipy_ls_lm")
    assert "'b':" in str(e.value)
    assert "c    inf" in str(e.value)


def test_non_finite_residuals_at_start_raise_for_derivative_free_algorithm():
    @mark.least_squares
    def fun(x):
        return np.array([x[0], np.inf if x[1] == 2 else x[1]])

    with pytest.raises(InvalidFunctionError, match="not finite"):
        minimize(fun, params=np.array([1.0, 2.0]), algorithm="scipy_neldermead")


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_non_finite_jac_at_start_raises(value):
    def jac(x):
        return np.array([value, 2 * x[1]])

    with pytest.raises(InvalidFunctionError, match="provided via jac is not finite"):
        minimize(lambda x: x @ x, params=np.ones(2), algorithm="scipy_lbfgsb", jac=jac)


def test_non_finite_fun_and_jac_at_start_raises():
    def fun_and_jac(x):
        return x["a"] ** 2 + x["b"] ** 2, {"a": 2 * x["a"], "b": np.nan}

    with pytest.raises(InvalidFunctionError, match="fun_and_jac is not finite") as e:
        minimize(
            fun_and_jac=fun_and_jac,
            params={"a": 1.0, "b": 2.0},
            algorithm="scipy_lbfgsb",
        )
    assert "'b': nan" in str(e.value)
