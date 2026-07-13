"""Tests for (almost) algorithm independent properties of maximize and minimize."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic import mark
from optimagic.batch_evaluators import joblib_batch_evaluator
from optimagic.examples.criterion_functions import sos_scalar
from optimagic.exceptions import InvalidFunctionError, InvalidNumdiffOptionsError
from optimagic.optimization.optimize import maximize, minimize


def test_minimize_uses_custom_batch_evaluator():
    """A custom batch_evaluator passed to minimize is the one optimagic calls."""
    used = []

    def spy_batch_evaluator(
        func, arguments, *, n_cores=1, error_handling="continue", unpack_symbol=None
    ):
        used.append(len(arguments))
        return joblib_batch_evaluator(
            func,
            arguments,
            n_cores=n_cores,
            error_handling=error_handling,
            unpack_symbol=unpack_symbol,
        )

    minimize(
        fun=mark.least_squares(lambda x: x),
        params=np.array([1.0, 2.0, 3.0]),
        algorithm="tranquilo_ls",
        batch_evaluator=spy_batch_evaluator,
        algo_options={"stopping_maxiter": 1},
    )

    assert used


def test_maximize_uses_custom_batch_evaluator():
    """Maximize threads a custom batch_evaluator through, same as minimize."""
    used = []

    def spy_batch_evaluator(
        func, arguments, *, n_cores=1, error_handling="continue", unpack_symbol=None
    ):
        used.append(len(arguments))
        return joblib_batch_evaluator(
            func,
            arguments,
            n_cores=n_cores,
            error_handling=error_handling,
            unpack_symbol=unpack_symbol,
        )

    maximize(
        fun=lambda x: -x @ x,
        params=np.array([1.0, 2.0, 3.0]),
        algorithm="tranquilo",
        batch_evaluator=spy_batch_evaluator,
        algo_options={"stopping_maxiter": 1},
    )

    assert used


def test_minimize_rejects_unknown_batch_evaluator():
    """An unknown batch_evaluator name raises a clear error."""
    with pytest.raises(ValueError, match="Invalid batch evaluator"):
        minimize(
            fun=mark.least_squares(lambda x: x),
            params=np.array([1.0, 2.0]),
            algorithm="tranquilo_ls",
            batch_evaluator="does_not_exist",
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
            numdiff_options={"bla": 15},
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


def test_fun_and_jac_list():
    with pytest.raises(NotImplementedError):
        minimize(
            fun_and_jac=[lambda x: (x @ x, 2 * x)],
            params=np.arange(5),
            algorithm="scipy_lbfgsb",
        )
