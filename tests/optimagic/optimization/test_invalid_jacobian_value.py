import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om
from optimagic.exceptions import UserFunctionRuntimeError
from optimagic.optimization.optimize import minimize

# ======================================================================================
# Test setup:
# --------------------------------------------------------------------------------------
# We test that minimize raises an error if the user function returns a jacobian
# containing invalid values (np.inf, np.nan). To test that this works not only at
# the start parameters, we create jac functions that return invalid values if the
# parameter norm becomes smaller than one.
# ======================================================================================


@pytest.fixture
def params():
    return {"a": 1, "b": np.array([3, 4])}


def sphere(params):
    return params["a"] ** 2 + (params["b"] ** 2).sum()


def sphere_gradient(params):
    return {
        "a": 2 * params["a"],
        "b": 2 * params["b"],
    }


def sphere_and_gradient(params):
    return sphere(params), sphere_gradient(params)


def params_norm(params):
    squared_norm = params["a"] ** 2 + np.linalg.norm(params["b"]) ** 2
    return np.sqrt(squared_norm)


def get_invalid_jac(invalid_jac_value):
    """Get function that returns invalid jac if the parameter norm < 1."""

    def jac(params):
        if params_norm(params) < 1:
            return invalid_jac_value
        else:
            return sphere_gradient(params)

    return jac


def get_invalid_fun_and_jac(invalid_jac_value):
    """Get function that returns invalid fun and jac if the parameter norm < 1."""

    def fun_and_jac(params):
        if params_norm(params) < 1:
            return sphere(params), invalid_jac_value
        else:
            return sphere_and_gradient(params)

    return fun_and_jac


INVALID_JACOBIAN_VALUES = [
    {"a": np.inf, "b": 2 * np.array([1, 2])},
    {"a": 1, "b": 2 * np.array([np.inf, 2])},
    {"a": np.nan, "b": 2 * np.array([1, 2])},
    {"a": 1, "b": 2 * np.array([np.nan, 2])},
]


# ======================================================================================
# Test Invalid Jacobian raises proper error with jac argument
# ======================================================================================


@pytest.mark.parametrize("invalid_jac_value", INVALID_JACOBIAN_VALUES)
def test_minimize_with_invalid_jac(invalid_jac_value, params):
    with pytest.raises(
        UserFunctionRuntimeError,
        match=(
            "The optimization failed because the derivative provided via jac "
            "contains infinite or NaN values."
        ),
    ):
        minimize(
            fun=sphere,
            params=params,
            algorithm="scipy_lbfgsb",
            jac=get_invalid_jac(invalid_jac_value),
        )


# ======================================================================================
# Test Invalid Jacobian raises proper error with fun_and_jac argument
# ======================================================================================


@pytest.mark.parametrize("invalid_jac_value", INVALID_JACOBIAN_VALUES)
def test_minimize_with_invalid_fun_and_jac(invalid_jac_value, params):
    with pytest.raises(
        UserFunctionRuntimeError,
        match=(
            "The optimization failed because the derivative provided via fun_and_jac "
            "contains infinite or NaN values."
        ),
    ):
        minimize(
            params=params,
            algorithm="scipy_lbfgsb",
            fun_and_jac=get_invalid_fun_and_jac(invalid_jac_value),
        )


# ======================================================================================
# Test that invalid derivatives with respect to fixed parameters are ignored
# ======================================================================================


def _fix_a():
    return om.FixedConstraint(selector=lambda params: params["a"])


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_minimize_ignores_invalid_jac_of_fixed_params(invalid_value, params):
    def jac(params):
        return {"a": invalid_value, "b": 2 * params["b"]}

    res = minimize(
        fun=sphere,
        params=params,
        algorithm="scipy_lbfgsb",
        jac=jac,
        constraints=_fix_a(),
    )

    assert res.params["a"] == 1
    aaae(res.params["b"], np.zeros(2))


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_minimize_ignores_invalid_fun_and_jac_of_fixed_params(invalid_value, params):
    def fun_and_jac(params):
        return sphere(params), {"a": invalid_value, "b": 2 * params["b"]}

    res = minimize(
        params=params,
        algorithm="scipy_lbfgsb",
        fun_and_jac=fun_and_jac,
        constraints=_fix_a(),
    )

    assert res.params["a"] == 1
    aaae(res.params["b"], np.zeros(2))


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_minimize_ignores_invalid_least_squares_jac_of_fixed_params(invalid_value):
    @om.mark.least_squares
    def fun(x):
        return x

    @om.mark.least_squares
    def jac(x):
        out = np.eye(len(x))
        out[:, 0] = invalid_value
        return out

    res = minimize(
        fun=fun,
        params=np.array([1.0, 2.0, 3.0]),
        algorithm="scipy_ls_lm",
        jac=jac,
        constraints=om.FixedConstraint(selector=lambda x: x[0]),
    )

    aaae(res.params, np.array([1.0, 0.0, 0.0]))


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_minimize_with_invalid_jac_of_free_params_and_fixed_params(
    invalid_value, params
):
    def jac(params):
        return {
            "a": 2 * params["a"],
            "b": np.array([invalid_value, 2 * params["b"][1]]),
        }

    with pytest.raises(UserFunctionRuntimeError, match="infinite or NaN values"):
        minimize(
            fun=sphere,
            params=params,
            algorithm="scipy_lbfgsb",
            jac=jac,
            constraints=_fix_a(),
        )
