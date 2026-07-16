"""Tests for (almost) algorithm independent properties of maximize and minimize."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic import Bounds, ScalingOptions, build_internal_fun, mark
from optimagic.batch_evaluators import joblib_batch_evaluator
from optimagic.examples.criterion_functions import sos_scalar
from optimagic.exceptions import InvalidFunctionError, InvalidNumdiffOptionsError
from optimagic.logging.types import IterationState
from optimagic.optimization.history import HistoryEntry
from optimagic.optimization.optimize import maximize, minimize


def test_build_internal_fun_evaluates_a_point_like_minimize():
    """build_internal_fun returns the per-point (value, history, log) evaluator.

    For a least-squares problem the internal value at a point is the residual vector,
    the history entry restores the external params, and a log row is produced — the
    same triple the driver's batch machinery consumes, so a distributed worker can
    evaluate the driver's broadcast points with an interchangeable callable.
    """
    evaluate = build_internal_fun(
        fun=mark.least_squares(lambda x: x),
        params=np.array([1.0, 2.0, 3.0]),
        algorithm="pounders",
    )

    value, hist_entry, log_entry = evaluate(np.array([4.0, 5.0, 6.0]))

    aaae(value, [4.0, 5.0, 6.0])
    assert isinstance(hist_entry, HistoryEntry)
    aaae(hist_entry.params, [4.0, 5.0, 6.0])
    assert isinstance(log_entry, IterationState)


def test_build_internal_fun_matches_the_driver_evaluator():
    """The worker evaluator equals the driver's, including tree reparametrization.

    minimize's batch machinery and build_internal_fun build the same converter from
    the same params, so for any internal point both map it to the same external params
    and return the same value — the property a driver/worker split relies on.
    """
    captured = {}

    def capturing_batch_evaluator(
        func, arguments, *, n_cores=1, error_handling="continue", unpack_symbol=None
    ):
        captured["driver_fun"] = func
        return joblib_batch_evaluator(
            func, arguments, n_cores=n_cores, error_handling=error_handling
        )

    fun = mark.least_squares(lambda p: p["x"])
    params = {"x": np.array([1.0, 2.0, 3.0])}

    minimize(
        fun=fun,
        params=params,
        algorithm="pounders",
        batch_evaluator=capturing_batch_evaluator,
        algo_options={"stopping_maxiter": 1, "n_cores": 2},
    )

    worker_fun = build_internal_fun(fun=fun, params=params, algorithm="pounders")

    point = np.array([0.5, -0.5, 0.25])
    aaae(worker_fun(point)[0], captured["driver_fun"](point)[0])


def test_build_internal_fun_matches_the_driver_evaluator_under_scaling():
    """With scaling, the worker evaluator still equals the driver's.

    Scaling changes the internal parameter space (here: bounds-scaling maps each
    parameter to [0, 1]), so a worker that builds its evaluator without the driver's
    scaling maps the same internal point to different external params. Passing the
    same `scaling` to build_internal_fun keeps the two callables interchangeable.
    """
    captured = {}

    def capturing_batch_evaluator(
        func, arguments, *, n_cores=1, error_handling="continue", unpack_symbol=None
    ):
        captured["driver_fun"] = func
        return joblib_batch_evaluator(
            func, arguments, n_cores=n_cores, error_handling=error_handling
        )

    fun = mark.least_squares(lambda p: p["x"])
    params = {"x": np.array([1.0, 2.0, 3.0])}
    bounds = Bounds(
        lower={"x": np.zeros(3)},
        upper={"x": np.full(3, 10.0)},
    )
    scaling = ScalingOptions(method="bounds")

    minimize(
        fun=fun,
        params=params,
        algorithm="pounders",
        bounds=bounds,
        scaling=scaling,
        batch_evaluator=capturing_batch_evaluator,
        algo_options={"stopping_maxiter": 1, "n_cores": 2},
    )

    worker_fun = build_internal_fun(
        fun=fun,
        params=params,
        algorithm="pounders",
        bounds=bounds,
        scaling=scaling,
    )

    point = np.array([0.25, 0.5, 0.75])
    driver_value, driver_hist, _ = captured["driver_fun"](point)
    worker_value, worker_hist, _ = worker_fun(point)
    aaae(worker_value, driver_value)
    aaae(worker_hist.params["x"], driver_hist.params["x"])


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
