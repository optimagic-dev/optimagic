import numpy as np
import pytest

from optimagic.exceptions import UserFunctionRuntimeError
from optimagic.optimization.optimize import minimize


def sphere(params):
    return (
        params["a"] ** 2
        + (params["b"] ** 2).sum()
        + params["c"]["x"] ** 2
        + params["c"]["y"] ** 2
    )


def sphere_gradient(params):
    return {
        "a": 2 * params["a"],
        "b": 2 * params["b"],
        "c": {"x": 2 * params["c"]["x"], "y": 2 * params["c"]["y"]},
    }


def sphere_and_gradient(params):
    return sphere(params), sphere_gradient(params)


def params_norm(params):
    squared_norm = (
        params["a"] ** 2
        + np.linalg.norm(params["b"]) ** 2
        + params["c"]["x"] ** 2
        + params["c"]["y"] ** 2
    )
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
    {"a": np.inf, "b": 2 * np.array([1, 2]), "c": {"x": 1, "y": 2}},
    {"a": 1, "b": 2 * np.array([np.inf, 2]), "c": {"x": 1, "y": 2}},
    {"a": np.nan, "b": 2 * np.array([1, 2]), "c": {"x": 1, "y": 2}},
    {"a": 1, "b": 2 * np.array([np.nan, 2]), "c": {"x": 1, "y": 2}},
    {"a": 1, "b": 2 * np.array([1, 2]), "c": {"x": np.inf, "y": 2}},
    {"a": 1, "b": 2 * np.array([1, 2]), "c": {"x": 1, "y": np.nan}},
]

PARAMS = {"a": 1, "b": np.array([3, 4]), "c": {"x": 5, "y": 6}}


@pytest.mark.parametrize("invalid_jac_value", INVALID_JACOBIAN_VALUES)
def test_minimize_with_invalid_jac(invalid_jac_value):
    with pytest.raises(
        UserFunctionRuntimeError,
        match=(
            "The optimization failed because the derivative provided via jac "
            "contains infinite or NaN values."
        ),
    ):
        minimize(
            fun=sphere,
            params=PARAMS,
            algorithm="scipy_lbfgsb",
            jac=get_invalid_jac(invalid_jac_value),
        )


@pytest.mark.parametrize("invalid_jac_value", INVALID_JACOBIAN_VALUES)
def test_minimize_with_invalid_fun_and_jac(invalid_jac_value):
    with pytest.raises(
        UserFunctionRuntimeError,
        match=(
            "The optimization failed because the derivative provided via fun_and_jac "
            "contains infinite or NaN values."
        ),
    ):
        minimize(
            params=PARAMS,
            algorithm="scipy_lbfgsb",
            fun_and_jac=get_invalid_fun_and_jac(invalid_jac_value),
        )
