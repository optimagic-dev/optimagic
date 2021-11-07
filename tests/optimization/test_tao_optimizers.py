"""Test the wrapper around pounders."""
import functools

import numpy as np
import pandas as pd
import pytest
from estimagic.config import IS_PETSC4PY_INSTALLED
from estimagic.optimization.optimize import minimize

if not IS_PETSC4PY_INSTALLED:
    pytestmark = pytest.mark.skip(reason="petsc4py is not installed.")


NUM_AGENTS = 2_000


def get_random_params(length, low=0, high=1, lower_bound=-np.inf, upper_bound=np.inf):
    params = pd.DataFrame(
        {
            "value": np.random.uniform(low, high, size=length),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
    )

    return params


def test_robustness():
    np.random.seed(5471)
    true_params = get_random_params(2)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = functools.partial(_ols_criterion, endog=endog, exog=exog)
    result = minimize(criterion_func, start_params, "tao_pounders")

    x = np.column_stack([np.ones_like(exog), exog])
    y = endog.reshape(len(endog), 1)
    expected = np.linalg.lstsq(x, y, rcond=None)[0].flatten()

    np.testing.assert_almost_equal(result["solution_x"], expected, decimal=6)


def test_box_constr():
    np.random.seed(5472)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)

    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = functools.partial(_ols_criterion, endog=endog, exog=exog)
    result = minimize(criterion_func, start_params, "tao_pounders")

    assert 0 <= result["solution_x"][0] <= 0.3
    assert 0 <= result["solution_x"][1] <= 0.3


def test_max_iters():
    np.random.seed(5473)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = functools.partial(_ols_criterion, endog=endog, exog=exog)
    result = minimize(
        criterion_func,
        start_params,
        "tao_pounders",
        algo_options={"stopping.max_iterations": 25},
    )

    assert result["message"] == "user defined" or result["message"] == "step size small"
    if result["convergence_code"] == 8:
        assert result["solution_criterion"][0] == 25


def test_grtol():
    np.random.seed(5474)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = functools.partial(_ols_criterion, endog=endog, exog=exog)
    result = minimize(
        criterion_func,
        start_params,
        "tao_pounders",
        algo_options={
            "convergence.absolute_gradient_tolerance": False,
            "convergence.scaled_gradient_tolerance": False,
        },
    )

    assert (
        result["message"] == "relative_gradient_tolerance below critical value"
        or result["message"] == "step size small"
    )

    if result["convergence_code"] == 4:
        assert result["solution_criterion"][2] / result["solution_criterion"][1] < 10


def test_gatol():
    np.random.seed(5475)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = functools.partial(_ols_criterion, endog=endog, exog=exog)
    result = minimize(
        criterion_func,
        start_params,
        "tao_pounders",
        algo_options={
            "convergence.relative_gradient_tolerance": False,
            "convergence.scaled_gradient_tolerance": False,
        },
    )

    assert (
        result["message"] == "absolute_gradient_tolerance below critical value"
        or result["message"] == "step size small"
    )
    if result["convergence_code"] == 3:
        assert result["solution_criterion"][2] < 1e-4


def test_gttol():
    np.random.seed(5476)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = functools.partial(_ols_criterion, endog=endog, exog=exog)
    result = minimize(
        criterion_func,
        start_params,
        "tao_pounders",
        algo_options={
            "convergence.relative_gradient_tolerance": False,
            "convergence.absolute_gradient_tolerance": False,
        },
    )

    assert (
        result["message"] == "gradient_total_tolerance below critical value"
        or result["message"] == "step size small"
    )

    if result["convergence_code"] == 5:
        assert result["solution_criterion"][2] < 1


def test_tol():
    np.random.seed(5477)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = functools.partial(_ols_criterion, endog=endog, exog=exog)
    result = minimize(
        criterion_func,
        start_params,
        "tao_pounders",
        algo_options={
            "convergence.absolute_gradient_tolerance": 1e-7,
            "convergence.relative_gradient_tolerance": 1e-7,
            "convergence.scaled_gradient_tolerance": 1e-9,
        },
    )

    if result["convergence_code"] == 3:
        assert result["solution_criterion"][2] < 0.00000001
    elif result["convergence_code"] == 4:
        assert (
            result["solution_criterion"][2] / result["solution_criterion"][1]
            < 0.00000001
        )


def _nonlinear_criterion(x, endog, exog):
    error = endog - np.exp(-x.loc[0, "value"] * exog) / (
        x.loc[1, "value"] + x.loc[2, "value"] * exog
    )
    return {
        "value": np.sum(np.square(error)),
        "root_contributions": error,
    }


def _ols_criterion(x, endog, exog):
    error = endog - x.loc[0, "value"] - x.loc[1, "value"] * exog
    return {
        "value": np.sum(np.square(error)),
        "root_contributions": error,
    }


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
