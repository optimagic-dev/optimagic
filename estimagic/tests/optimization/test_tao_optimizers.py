"""Test the wrapper around pounders."""
import functools
import sys

import numpy as np
import pandas as pd
import pytest

from estimagic.optimization.tao_optimizers import tao_pounders

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="TAO is not available on Windows."
)


NUM_AGENTS = 2_000


def test_tao_not_available_on_windows(monkeypatch):
    monkeypatch.setattr("estimagic.optimization.tao_optimizers.sys.platform", "win32")
    with pytest.raises(NotImplementedError):
        tao_pounders(None, None, None, None)


def get_random_params(length, low=0, high=1, lower_bound=-np.inf, upper_bound=np.inf):
    params = pd.DataFrame(
        {
            "value": np.random.uniform(low, high, size=length),
            "lower": lower_bound,
            "upper": upper_bound,
        }
    )

    return params


def _make_tao_criterion_function(endog, exog, kind):
    criterion_func = _ols_criterion if kind == "ols" else _nonlinear_criterion
    criterion_func = functools.partial(criterion_func, endog=endog, exog=exog)

    def _wrapper_criterion_and_derivative(
        x, task=None, algorithm_info=None, first_criterion_evaluation=None
    ):
        return criterion_func(x=x)

    return functools.partial(
        _wrapper_criterion_and_derivative,
        first_criterion_evaluation={
            "output": {"root_contributions": [None] * len(endog)}
        },
    )


def test_robustness():
    np.random.seed(5471)
    true_params = get_random_params(2)
    start_params = get_random_params(2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = _make_tao_criterion_function(endog, exog, "ols")
    result = tao_pounders(objective, start_params["value"].to_numpy(), *bounds)

    x = np.column_stack([np.ones_like(exog), exog])
    y = endog.reshape(len(endog), 1)
    expected = np.linalg.lstsq(x, y, rcond=None)[0].flatten()

    np.testing.assert_almost_equal(result["solution_x"], expected, decimal=6)


def test_box_constr():
    np.random.seed(5472)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = _make_tao_criterion_function(endog, exog, "ols")
    result = tao_pounders(objective, start_params["value"].to_numpy(), *bounds)

    assert 0 <= result["solution_x"][0] <= 0.3
    assert 0 <= result["solution_x"][1] <= 0.3


def test_max_iters():
    np.random.seed(5473)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = get_random_params(2, 0.1, 0.2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = _make_tao_criterion_function(endog, exog, "ols")
    result = tao_pounders(
        objective, start_params["value"].to_numpy(), *bounds, max_iterations=25
    )

    assert result["message"] == "user defined" or result["message"] == "step size small"
    if result["convergence_code"] == 8:
        assert result["solution_criterion"][0] == 25


def test_grtol():
    np.random.seed(5474)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = get_random_params(2, 0.1, 0.2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = _make_tao_criterion_function(endog, exog, "ols")
    result = tao_pounders(
        objective,
        start_params["value"].to_numpy(),
        *bounds,
        gradient_absolute_tolerance=False,
        gradient_total_tolerance=False,
    )

    assert (
        result["message"] == "gradient_relative_tolerance below critical value"
        or result["message"] == "step size small"
    )

    if result["convergence_code"] == 4:
        assert result["solution_criterion"][2] / result["solution_criterion"][1] < 10


def test_gatol():
    np.random.seed(5475)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = get_random_params(2, 0.1, 0.2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = _make_tao_criterion_function(endog, exog, "ols")
    calculated = tao_pounders(
        objective,
        start_params["value"].to_numpy(),
        *bounds,
        gradient_relative_tolerance=False,
        gradient_total_tolerance=False,
    )

    assert (
        calculated["message"] == "gradient_absolute_tolerance below critical value"
        or calculated["message"] == "step size small"
    )
    if calculated["convergence_code"] == 3:
        assert calculated["solution_criterion"][2] < 1e-4


def test_gttol():
    np.random.seed(5476)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = get_random_params(2, 0.1, 0.2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = _make_tao_criterion_function(endog, exog, "ols")
    calculated = tao_pounders(
        objective,
        start_params["value"].to_numpy(),
        *bounds,
        gradient_relative_tolerance=False,
        gradient_absolute_tolerance=False,
    )

    assert (
        calculated["message"] == "gradient_total_tolerance below critical value"
        or calculated["message"] == "step size small"
    )

    if calculated["convergence_code"] == 5:
        assert calculated["solution_criterion"][2] < 1


def test_tol():
    np.random.seed(5477)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = get_random_params(2, 0.1, 0.2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = _make_tao_criterion_function(endog, exog, "ols")
    calculated = tao_pounders(
        objective,
        start_params["value"].to_numpy(),
        *bounds,
        gradient_absolute_tolerance=1e-7,
        gradient_relative_tolerance=1e-7,
        gradient_total_tolerance=1e-9,
    )

    if calculated["convergence_code"] == 3:
        assert calculated["solution_criterion"][2] < 0.00000001
    elif calculated["convergence_code"] == 4:
        assert (
            calculated["solution_criterion"][2] / calculated["solution_criterion"][1]
            < 0.00000001
        )


def _nonlinear_criterion(endog, exog, x):
    return endog - np.exp(-x[0] * exog) / (x[1] + x[2] * exog)


def _ols_criterion(endog, exog, x):
    return endog - x[0] - x[1] * exog


def _simulate_sample(num_agents, paras, error_term_high=0.5):
    exog = np.random.uniform(0, 1, num_agents)
    error_term = np.random.normal(0, error_term_high, num_agents)
    endog = (
        np.exp(-paras.at[0, "value"] * exog)
        / (paras.at[1, "value"] + paras.at[2, "value"] * exog)
        + error_term
    )

    return exog, endog


def _simulate_ols_sample(num_agents, paras):
    exog = np.random.uniform(-5, 5, num_agents)
    error_term = np.random.normal(0, 1, num_agents)
    endog = paras.at[0, "value"] + paras.at[1, "value"] * exog + error_term

    return exog, endog
