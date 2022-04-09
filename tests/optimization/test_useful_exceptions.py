import numpy as np
import pandas as pd
import pytest
from estimagic.exceptions import InvalidKwargsError
from estimagic.optimization.optimize import minimize


def just_fail(params):
    raise RuntimeError()


def test_missing_criterion_kwargs():
    def f(params, bla, blubb):
        return (params["value"].to_numpy() ** 2).sum()

    params = pd.DataFrame(np.ones((3, 1)), columns=["value"])

    with pytest.raises(InvalidKwargsError):
        minimize(f, params, "scipy_lbfgsb", criterion_kwargs={"bla": 3})


def test_missing_derivative_kwargs():
    def f(params):
        return (params["value"].to_numpy() ** 2).sum()

    def grad(params, bla, blubb):
        return params["value"].to_numpy() * 2

    params = pd.DataFrame(np.ones((3, 1)), columns=["value"])

    with pytest.raises(InvalidKwargsError):
        minimize(
            f, params, "scipy_lbfgsb", derivative=grad, derivative_kwargs={"bla": 3}
        )


def test_missing_criterion_and_derivative_kwargs():
    def f(params):
        return (params["value"].to_numpy() ** 2).sum()

    def f_and_grad(params, bla, blubb):
        return f(params), params["value"].to_numpy() * 2

    params = pd.DataFrame(np.ones((3, 1)), columns=["value"])

    with pytest.raises(InvalidKwargsError):
        minimize(
            f,
            params,
            "scipy_lbfgsb",
            criterion_and_derivative=f_and_grad,
            criterion_and_derivative_kwargs={"bla": 3},
        )
