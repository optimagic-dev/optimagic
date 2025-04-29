import numpy as np
import pandas as pd
import pytest

from optimagic.exceptions import UserFunctionRuntimeError
from optimagic.optimization.optimize import minimize


def test_with_infinite_jac_value_unconditional_in_lists():
    def sphere(params):
        return params @ params

    def sphere_gradient(params):
        return np.full_like(params, np.inf)

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun=sphere,
            params=np.arange(10) + 400,
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )


def test_with_infinite_jac_value_conditional_in_lists():
    def sphere(params):
        return params @ params

    def true_gradient(params):
        return 2 * params

    def param_norm(params):
        return np.norm(params)

    def sphere_gradient(params):
        if param_norm(params) >= 1:
            return true_gradient(params)
        else:
            return np.full_like(params, np.inf)

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun=sphere,
            params=np.arange(10) + 400,
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )


def test_with_infinite_fun_and_jac_value_unconditional_in_lists():
    def sphere_and_gradient(params):
        function_value = params @ params
        grad = np.full_like(params, np.inf)
        return function_value, grad

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun_and_jac=sphere_and_gradient,
            params=np.arange(10) + 400,
            algorithm="scipy_lbfgsb",
        )


def test_with_infinite_fun_and_jac_value_conditional_in_lists():
    def true_gradient(params):
        return 2 * params

    def param_norm(params):
        return np.norm(params)

    def sphere_gradient(params):
        if param_norm(params) >= 1:
            return true_gradient(params)
        else:
            return np.full_like(params, np.inf)

    def sphere_and_gradient(params):
        function_value = params @ params
        grad = sphere_gradient(params)
        return function_value, grad

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun_and_jac=sphere_and_gradient,
            params=np.arange(10) + 400,
            algorithm="scipy_lbfgsb",
        )


def test_with_infinite_jac_value_unconditional_in_dicts():
    def sphere(params):
        return params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()

    def sphere_gradient(params):
        return {"a": np.inf, "b": np.inf, "c": np.full_like(params["c"], np.inf)}

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun=sphere,
            params={"a": 400, "b": 400, "c": pd.Series([200, 300, 400])},
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )


def test_with_infinite_jac_value_conditional_in_dicts():
    def sphere(params):
        return params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()

    def true_gradient(params):
        return {"a": 2 * params["a"], "b": 2 * params["b"], "c": 2 * params["c"]}

    def param_norm(params):
        squared_norm = (
            params["a"] ** 2 + params["b"] ** 2 + np.linalg.norm(params["c"]) ** 2
        )
        return np.sqrt(squared_norm)

    def sphere_gradient(params):
        if param_norm(params) >= 1:
            return true_gradient(params)
        else:
            return {"a": np.inf, "b": np.inf, "c": np.full_like(params["c"], np.inf)}

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun=sphere,
            params={"a": 400, "b": 400, "c": pd.Series([200, 300, 400])},
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )


def test_with_infinite_fun_and_jac_value_unconditional_in_dicts():
    def sphere_and_gradient(params):
        function_value = params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()
        grad = {"a": np.inf, "b": np.inf, "c": np.full_like(params["c"], np.inf)}
        return function_value, grad

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun_and_jac=sphere_and_gradient,
            params={"a": 400, "b": 400, "c": pd.Series([200, 300, 400])},
            algorithm="scipy_lbfgsb",
        )


def test_with_infinite_fun_and_jac_value_conditional_in_dicts():
    def true_gradient(params):
        return {"a": 2 * params["a"], "b": 2 * params["b"], "c": 2 * params["c"]}

    def param_norm(params):
        squared_norm = (
            params["a"] ** 2 + params["b"] ** 2 + np.linalg.norm(params["c"]) ** 2
        )
        return np.sqrt(squared_norm)

    def sphere_gradient(params):
        if param_norm(params) >= 1:
            return true_gradient(params)
        else:
            return {"a": np.inf, "b": np.inf, "c": np.full_like(params["c"], np.inf)}

    def sphere_and_gradient(params):
        function_value = params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()
        grad = sphere_gradient(params)
        return function_value, grad

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun_and_jac=sphere_and_gradient,
            params={"a": 400, "b": 400, "c": pd.Series([200, 300, 400])},
            algorithm="scipy_lbfgsb",
        )


def test_with_nan_jac_value_unconditional_in_lists():
    def sphere(params):
        return params @ params

    def sphere_gradient(params):
        return np.full_like(params, np.nan)

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun=sphere,
            params=np.arange(10) + 400,
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )


def test_with_nan_jac_value_conditional_in_lists():
    def sphere(params):
        return params @ params

    def true_gradient(params):
        return 2 * params

    def param_norm(params):
        return np.norm(params)

    def sphere_gradient(params):
        if param_norm(params) >= 1:
            return true_gradient(params)
        else:
            return np.full_like(params, np.nan)

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun=sphere,
            params=np.arange(10) + 400,
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )


def test_with_nan_fun_and_jac_value_unconditional_in_lists():
    def sphere_and_gradient(params):
        function_value = params @ params
        grad = np.full_like(params, np.nan)
        return function_value, grad

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun_and_jac=sphere_and_gradient,
            params=np.arange(10) + 400,
            algorithm="scipy_lbfgsb",
        )


def test_with_nan_fun_and_jac_value_conditional_in_lists():
    def true_gradient(params):
        return 2 * params

    def param_norm(params):
        return np.norm(params)

    def sphere_gradient(params):
        if param_norm(params) >= 1:
            return true_gradient(params)
        else:
            return np.full_like(params, np.nan)

    def sphere_and_gradient(params):
        function_value = params @ params
        grad = sphere_gradient(params)
        return function_value, grad

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun_and_jac=sphere_and_gradient,
            params=np.arange(10) + 400,
            algorithm="scipy_lbfgsb",
        )


def test_with_nan_jac_value_unconditional_in_dicts():
    def sphere(params):
        return params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()

    def sphere_gradient(params):
        return {"a": np.nan, "b": np.nan, "c": np.full_like(params["c"], np.nan)}

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun=sphere,
            params={"a": 400, "b": 400, "c": pd.Series([200, 300, 400])},
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )


def test_with_nan_jac_value_conditional_in_dicts():
    def sphere(params):
        return params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()

    def true_gradient(params):
        return {"a": 2 * params["a"], "b": 2 * params["b"], "c": 2 * params["c"]}

    def param_norm(params):
        squared_norm = (
            params["a"] ** 2 + params["b"] ** 2 + np.linalg.norm(params["c"]) ** 2
        )
        return np.sqrt(squared_norm)

    def sphere_gradient(params):
        if param_norm(params) >= 1:
            return true_gradient(params)
        else:
            return {"a": np.nan, "b": np.nan, "c": np.full_like(params["c"], np.nan)}

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun=sphere,
            params={"a": 400, "b": 400, "c": pd.Series([200, 300, 400])},
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )


def test_with_nan_fun_and_jac_value_unconditional_in_dicts():
    def sphere_and_gradient(params):
        function_value = params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()
        grad = {"a": np.nan, "b": np.nan, "c": np.full_like(params["c"], np.nan)}
        return function_value, grad

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun_and_jac=sphere_and_gradient,
            params={"a": 400, "b": 400, "c": pd.Series([200, 300, 400])},
            algorithm="scipy_lbfgsb",
        )


def test_with_nan_fun_and_jac_value_conditional_in_dicts():
    def true_gradient(params):
        return {"a": 2 * params["a"], "b": 2 * params["b"], "c": 2 * params["c"]}

    def param_norm(params):
        squared_norm = (
            params["a"] ** 2 + params["b"] ** 2 + np.linalg.norm(params["c"]) ** 2
        )
        return np.sqrt(squared_norm)

    def sphere_gradient(params):
        if param_norm(params) >= 1:
            return true_gradient(params)
        else:
            return {"a": np.nan, "b": np.nan, "c": np.full_like(params["c"], np.nan)}

    def sphere_and_gradient(params):
        function_value = params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()
        grad = sphere_gradient(params)
        return function_value, grad

    with pytest.raises(UserFunctionRuntimeError):
        minimize(
            fun_and_jac=sphere_and_gradient,
            params={"a": 400, "b": 400, "c": pd.Series([200, 300, 400])},
            algorithm="scipy_lbfgsb",
        )
