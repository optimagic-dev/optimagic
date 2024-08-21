"""Test the wrapper around pounders."""

import functools

import numpy as np
import pandas as pd
import pytest
from optimagic.config import IS_PETSC4PY_INSTALLED
from optimagic.optimization.optimize import minimize
from optimagic.utilities import get_rng

if not IS_PETSC4PY_INSTALLED:
    pytestmark = pytest.mark.skip(reason="petsc4py is not installed.")


NUM_AGENTS = 2_000
from optimagic import mark


def get_random_params(
    length,
    rng,  # noqa: ARG001
    low=0,
    high=1,
    lower_bound=-np.inf,
    upper_bound=np.inf,
):
    params = pd.DataFrame(
        {
            "value": np.random.uniform(low, high, size=length),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
    )

    return params


def test_robustness():
    rng = get_rng(5471)
    true_params = get_random_params(2, rng)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, rng)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = mark.least_squares(
        functools.partial(_ols_criterion, endog=endog, exog=exog)
    )
    result = minimize(criterion_func, start_params, "tao_pounders")

    x = np.column_stack([np.ones_like(exog), exog])
    y = endog.reshape(len(endog), 1)
    expected = np.linalg.lstsq(x, y, rcond=None)[0].flatten()

    np.testing.assert_almost_equal(
        result.params["value"].to_numpy(), expected, decimal=6
    )


def test_box_constr():
    rng = get_rng(5472)
    true_params = get_random_params(2, rng, 0.3, 0.4, 0, 0.3)

    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, rng, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = mark.least_squares(
        functools.partial(_ols_criterion, endog=endog, exog=exog)
    )
    result = minimize(criterion_func, start_params, "tao_pounders")

    assert 0 <= result.params["value"].to_numpy()[0] <= 0.3
    assert 0 <= result.params["value"].to_numpy()[1] <= 0.3


def test_max_iters():
    rng = get_rng(5473)
    true_params = get_random_params(2, rng, 0.3, 0.4, 0, 0.3)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, rng, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = mark.least_squares(
        functools.partial(_ols_criterion, endog=endog, exog=exog)
    )
    result = minimize(
        criterion_func,
        start_params,
        "tao_pounders",
        algo_options={"stopping.maxiter": 25},
    )

    assert result.message in ("user defined", "step size small")


def test_grtol():
    rng = get_rng(5474)
    true_params = get_random_params(2, rng, 0.3, 0.4, 0, 0.3)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, rng, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = mark.least_squares(
        functools.partial(_ols_criterion, endog=endog, exog=exog)
    )
    result = minimize(
        criterion_func,
        start_params,
        "tao_pounders",
        algo_options={
            "convergence.gtol_abs": False,
            "convergence.gtol_scaled": False,
        },
    )

    assert result.message in (
        "relative_gradient_tolerance below critical value",
        "step size small",
    )


def test_gatol():
    rng = get_rng(5475)
    true_params = get_random_params(2, rng, 0.3, 0.4, 0, 0.3)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, rng, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = mark.least_squares(
        functools.partial(_ols_criterion, endog=endog, exog=exog)
    )
    result = minimize(
        criterion_func,
        start_params,
        "tao_pounders",
        algo_options={
            "convergence.gtol_rel": False,
            "convergence.gtol_scaled": False,
        },
    )

    assert result.message in (
        "absolute_gradient_tolerance below critical value",
        "step size small",
    )


def test_gttol():
    rng = get_rng(5476)
    true_params = get_random_params(2, rng, 0.3, 0.4, 0, 0.3)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, rng, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = mark.least_squares(
        functools.partial(_ols_criterion, endog=endog, exog=exog)
    )
    result = minimize(
        criterion_func,
        start_params,
        "tao_pounders",
        algo_options={
            "convergence.gtol_rel": False,
            "convergence.gtol_abs": False,
        },
    )

    assert result.message in (
        "gradient_total_tolerance below critical value",
        "step size small",
    )


def test_tol():
    rng = get_rng(5477)
    true_params = get_random_params(2, rng, 0.3, 0.4, 0, 0.3)
    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, rng, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    criterion_func = mark.least_squares(
        functools.partial(_ols_criterion, endog=endog, exog=exog)
    )
    minimize(
        criterion_func,
        start_params,
        "tao_pounders",
        algo_options={
            "convergence.gtol_abs": 1e-7,
            "convergence.gtol_rel": 1e-7,
            "convergence.gtol_scaled": 1e-9,
        },
    )


def _ols_criterion(x, endog, exog):
    return endog - x.loc[0, "value"] - x.loc[1, "value"] * exog


def _simulate_ols_sample(num_agents, paras):
    rng = get_rng(seed=1234)
    exog = rng.uniform(-5, 5, num_agents)
    error_term = rng.normal(0, 1, num_agents)
    endog = paras.at[0, "value"] + paras.at[1, "value"] * exog + error_term

    return exog, endog
