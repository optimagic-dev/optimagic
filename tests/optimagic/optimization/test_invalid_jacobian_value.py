import numpy as np
import pandas as pd
import pytest

from optimagic.exceptions import InvalidFunctionError
from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
)
from optimagic.optimization.optimize import minimize

SCALAR_VALUES = [
    ScalarFunctionValue(5),
]

LS_VALUES = [
    LeastSquaresFunctionValue(np.array([1, 2])),
    LeastSquaresFunctionValue({"a": 1, "b": 2}),
]

LIKELIHOOD_VALUES = [
    LikelihoodFunctionValue(np.array([1, 4])),
    LikelihoodFunctionValue({"a": 1, "b": 4}),
]


def test_with_infinite_jacobian_value_in_lists():
    def sphere(params):
        return params @ params

    def sphere_gradient(params):
        grad = 2 * params
        grad[(abs(grad) < 1.0) & (abs(grad) > 0.0)] = (
            np.sign(grad)[(abs(grad) < 1.0) & (abs(grad) > 0.0)] * np.inf
        )
        return grad

    with pytest.raises(InvalidFunctionError):
        minimize(
            fun=sphere,
            params=np.arange(10) + 400,
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )


def test_with_infinite_jacobian_value_in_dicts():
    def sphere(params):
        return params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()

    def sphere_gradient(params):
        grad = {
            "a": 2 * params["a"]
            if not ((abs(params["a"]) < 1.0) & (abs(params["a"]) > 0.0))
            else np.sign(params["a"]) * np.inf,
            "b": 2 * params["b"]
            if not ((abs(params["b"]) < 1.0) & (abs(params["b"]) > 0.0))
            else np.sign(params["b"]) * np.inf,
            "c": 2 * params["c"]
            if not ((abs(params["c"].sum()) < 1.0) & (abs(params["c"].sum()) > 0.0))
            else np.sign(params["c"]) * np.inf,
        }
        return grad

    with pytest.raises(InvalidFunctionError):
        minimize(
            fun=sphere,
            params={"a": 400, "b": 400, "c": pd.Series([200, 300, 400])},
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )


def test_with_nan_jacobian_value_in_lists():
    def sphere(params):
        return params @ params

    def sphere_gradient(params):
        grad = 2 * params
        grad[(abs(grad) < 1.0) & (abs(grad) > 0.0)] = (
            np.sign(grad)[(abs(grad) < 1.0) & (abs(grad) > 0.0)] * np.nan
        )
        return grad

    with pytest.raises(InvalidFunctionError):
        minimize(
            fun=sphere,
            params=np.arange(10) + 400,
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )


def test_with_nan_jacobian_value_in_dicts():
    def sphere(params):
        return params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()

    def sphere_gradient(params):
        grad = {
            "a": 2 * params["a"]
            if not ((abs(params["a"]) < 1.0) & (abs(params["a"]) > 0.0))
            else np.sign(params["a"]) * np.nan,
            "b": 2 * params["b"]
            if not ((abs(params["b"]) < 1.0) & (abs(params["b"]) > 0.0))
            else np.sign(params["b"]) * np.nan,
            "c": 2 * params["c"]
            if not ((abs(params["c"].sum()) < 1.0) & (abs(params["c"].sum()) > 0.0))
            else np.sign(params["c"]) * np.nan,
        }
        return grad

    with pytest.raises(InvalidFunctionError):
        minimize(
            fun=sphere,
            params={"a": 400, "b": 400, "c": pd.Series([200, 300, 400])},
            algorithm="scipy_lbfgsb",
            jac=sphere_gradient,
        )
