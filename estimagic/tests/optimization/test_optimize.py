"""Tests for (almost) algorithm independent properties of maximize and minimize."""
import numpy as np
import pandas as pd
import pytest

from estimagic.examples.criterion_functions import sos_scalar_criterion
from estimagic.optimization.optimize import maximize
from estimagic.optimization.optimize import minimize


def test_sign_is_switched_back_after_maximization():
    params = pd.DataFrame()
    params["value"] = [1, 2, 3]
    res = maximize(
        lambda params: 1 - params["value"] @ params["value"],
        params=params,
        algorithm="scipy_lbfgsb",
    )

    assert np.allclose(res["solution_criterion"], 1)


def test_warnings_with_old_bounds_names():
    base_params = pd.DataFrame()
    base_params["value"] = [1, 2, 3]

    for wrong_name in "lower", "upper":
        params = base_params.copy()
        params[wrong_name] = 0
        with pytest.warns(UserWarning):
            maximize(
                lambda params: 1 - params["value"] @ params["value"],
                params=params,
                algorithm="scipy_lbfgsb",
            )


def test_scipy_lbfgsb_actually_calls_criterion_and_derivative():
    params = pd.DataFrame(data=np.ones((10, 1)), columns=["value"])

    def raising_crit_and_deriv(params):
        raise Exception()

    with pytest.raises(Exception):
        minimize(
            criterion=sos_scalar_criterion,
            params=params,
            algorithm="scipy_lbfgsb",
            criterion_and_derivative=raising_crit_and_deriv,
        )
